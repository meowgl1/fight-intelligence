from __future__ import annotations

import cv2
import numpy as np


def _wrist_patch(frame, kp: list[float], size: int = 25):
    # Ritaglia una patch quadrata attorno al polso.
    # Ritorna None se la confidence è troppo bassa o se il punto è fuori frame.
    x, y, conf = kp
    if conf < 0.5:
        return None
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x) - size), max(0, int(y) - size)
    x2, y2 = min(w, int(x) + size), min(h, int(y) + size)
    patch = frame[y1:y2, x1:x2]
    return patch if patch.size > 0 else None


def _color_hist(patch) -> np.ndarray:
    # Istogramma del canale H in HSV (16 bin, range 0-180).
    # Il canale H cattura il colore del guanto senza essere sensibile alla luminosità.
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    norm = hist.sum()
    return hist / norm if norm > 0 else hist


def _similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    # Correlazione tra due istogrammi normalizzati (1.0 = identici).
    return float(cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_CORREL))


class GloveReID:
    # Mappa gli ID instabili di ByteTrack a due ID canonici stabili (1 e 2),
    # usando il colore dei guanti (keypoints polso COCO 9 e 10) come firma.
    #
    # Logica:
    #   1. Nei primi frame (wrist conf > 0.5) costruisce una firma colore per fighter.
    #   2. Quando ByteTrack genera un nuovo ID, lo abbina alla firma più simile.
    #   3. Se il colore non è disponibile, fallback sulla posizione bbox.
    #   4. Dedup per frame: due detection non possono ricevere lo stesso canonical ID.

    def __init__(self, match_threshold: float = 0.5, ema_window: int = 30):
        # match_threshold: correlazione minima per accettare un match colore.
        # ema_window: quanti campioni pesare nella media mobile della firma.
        self._signatures: dict[int, np.ndarray] = {}   # canonical_id → istogramma
        self._sig_counts: dict[int, int] = {}
        self._id_map: dict[int, int] = {}              # bytetrack_id → canonical_id
        self._last_bbox: dict[int, list[float]] = {}   # canonical_id → ultimo bbox
        self._match_threshold = match_threshold
        self._ema_window = ema_window

    def assign(self, frame, fighters: list[dict]) -> list[dict]:
        # Riceve la lista di fighter con ID ByteTrack e ritorna la stessa lista
        # con fighter_id rimappato su 1 o 2 (ID canonici stabili).
        used_canonical: set[int] = set()
        result = []

        for f in fighters:
            bid = f["fighter_id"]
            color = self._sample_glove(frame, f["keypoints"])

            if bid in self._id_map:
                cid = self._id_map[bid]
                # Aggiorna la firma anche per ID già noti (migliora nel tempo).
                if color is not None and cid not in used_canonical:
                    self._update_sig(cid, color)
            else:
                cid = self._resolve(bid, color, f["bbox"], used_canonical)

            # Dedup: se questo canonical è già stato assegnato in questo frame,
            # prendi l'altro (in un bout a 2 fighter non ci sono altre opzioni).
            if cid in used_canonical:
                others = [c for c in self._signatures if c not in used_canonical]
                cid = others[0] if others else cid

            used_canonical.add(cid)
            self._last_bbox[cid] = f["bbox"]
            result.append({**f, "fighter_id": cid})

        return result

    def _sample_glove(self, frame, kps: list) -> np.ndarray | None:
        # Prova polso sinistro (9) poi destro (10), ritorna il primo istogramma valido.
        for idx in (9, 10):
            patch = _wrist_patch(frame, kps[idx])
            if patch is not None:
                return _color_hist(patch)
        return None

    def _update_sig(self, cid: int, color: np.ndarray) -> None:
        # Media mobile esponenziale della firma colore.
        n = self._sig_counts.get(cid, 0)
        if cid not in self._signatures:
            self._signatures[cid] = color
        else:
            alpha = 1.0 / min(n + 1, self._ema_window)
            self._signatures[cid] = (1 - alpha) * self._signatures[cid] + alpha * color
        self._sig_counts[cid] = n + 1

    def _resolve(self, bid: int, color: np.ndarray | None, bbox: list[float], used: set[int]) -> int:
        # Assegna un nuovo bytetrack ID a un canonical ID.
        if len(self._signatures) < 2:
            # Primi fighter che vediamo: crea canonical ID sequenziale.
            cid = len(self._signatures) + 1
            if color is not None:
                self._update_sig(cid, color)
            self._id_map[bid] = cid
            return cid

        # Abbiamo già 2 canonical: matcha per colore se disponibile.
        if color is not None:
            candidates = [(cid, _similarity(color, sig))
                          for cid, sig in self._signatures.items()
                          if cid not in used]
            if candidates:
                best_cid, best_score = max(candidates, key=lambda x: x[1])
                if best_score >= self._match_threshold:
                    self._id_map[bid] = best_cid
                    self._update_sig(best_cid, color)
                    return best_cid

        # Fallback: canonical con bbox più vicino (centro).
        cid = self._nearest(bbox, used)
        self._id_map[bid] = cid
        return cid

    def _nearest(self, bbox: list[float], exclude: set[int]) -> int:
        # Ritorna il canonical ID il cui ultimo bbox ha il centro più vicino.
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        best_cid, best_dist = 1, float("inf")
        for cid, last in self._last_bbox.items():
            if cid in exclude:
                continue
            lx, ly = (last[0] + last[2]) / 2, (last[1] + last[3]) / 2
            dist = (cx - lx) ** 2 + (cy - ly) ** 2
            if dist < best_dist:
                best_dist, best_cid = dist, cid
        return best_cid
