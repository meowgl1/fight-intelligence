from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import yaml

from events.schema import StrikeEvent, StrikeType


def load_config(yaml_path: str) -> dict:
    # Carica il file YAML delle soglie. Se il file non esiste, lancia un errore
    # chiaro invece di andare avanti con valori sbagliati.
    p = Path(yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"Config non trovato: {yaml_path}")
    with open(p) as f:
        return yaml.safe_load(f)


def _angle_degrees(a: list[float], b: list[float], c: list[float]) -> float:
    # Calcola l'angolo in gradi al punto B, formato dai segmenti BA e BC.
    # Serve per misurare quanto è piegato un gomito (angolo spalla-gomito-polso).
    ax, ay = a[0] - b[0], a[1] - b[1]
    cx, cy = c[0] - b[0], c[1] - b[1]
    dot = ax * cx + ay * cy
    mag = math.hypot(ax, ay) * math.hypot(cx, cy)
    if mag == 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


def _bbox_height(bbox: list[float]) -> float:
    # Altezza del bounding box in pixel — serve per normalizzare le velocità.
    # Più il fighter è lontano, più piccolo è il bbox: la normalizzazione
    # rende le soglie indipendenti dalla distanza dalla telecamera.
    return max(bbox[3] - bbox[1], 1.0)


class HeuristicClassifier:
    # Classifica il tipo di colpo osservando come si muovono i keypoint nel tempo.
    # Per ogni fighter mantiene uno storico delle posizioni degli ultimi N frame.
    # Quando la velocità di un keypoint supera la soglia, tentiamo la classificazione.

    def __init__(self, config: dict):
        self.config = config
        self.window = config.get("velocity_window_frames", 5)
        self.min_kp_conf = config.get("min_confidence_keypoint", 0.5)
        # history: fighter_id → deque di snapshot keypoint [[x,y,conf]*17]
        self._history: dict[int, deque] = {}
        # ponytail: debounce semplice — un colpo per fighter ogni ~0.4s
        self._last_strike_frame: dict[int, int] = {}

    def update(self, fighter_id: int, keypoints: list[list[float]], bbox: list[float], frame_number: int, fps: float) -> StrikeEvent | None:
        # Aggiorna lo storico del fighter e controlla se c'è un colpo.
        if fighter_id not in self._history:
            self._history[fighter_id] = deque(maxlen=self.window)

        self._history[fighter_id].append(keypoints)

        # Serve almeno 2 frame per calcolare una velocità.
        if len(self._history[fighter_id]) < 2:
            return None

        cooldown = max(1, int(fps * 0.4))
        if frame_number - self._last_strike_frame.get(fighter_id, -cooldown) < cooldown:
            return None

        return self._classify(fighter_id, bbox, frame_number, fps)

    def _classify(self, fighter_id: int, bbox: list[float], frame_number: int, fps: float) -> StrikeEvent | None:
        # Proviamo ogni tipo di colpo in ordine di precedenza.
        # L'ordine conta: se un movimento soddisfa più criteri, vince il primo.
        h = self._history[fighter_id]
        norm = _bbox_height(bbox)
        thresholds = self.config.get("thresholds", {})

        for strike_name, cfg in thresholds.items():
            kp_id = cfg.get("keypoint_id", 10)
            min_v = cfg.get("min_velocity", 0.15)

            # Saltiamo se il keypoint trigger ha confidence troppo bassa.
            current_kp = h[-1][kp_id]
            if current_kp[2] < self.min_kp_conf:
                continue

            prev_kp = h[0][kp_id]
            if prev_kp[2] < self.min_kp_conf:
                continue

            # Velocità normalizzata = distanza percorsa in N frame / altezza bbox.
            dx = current_kp[0] - prev_kp[0]
            dy = current_kp[1] - prev_kp[1]
            velocity = math.hypot(dx, dy) / norm

            if velocity < min_v:
                continue

            # Controlli aggiuntivi per distinguere colpi con velocity simile.
            if not self._passes_shape_check(strike_name, cfg, h, kp_id):
                continue

            # Confidence dell'euristica: quanto supera la soglia minima.
            # Non è una probabilità ML — è un indicatore di "quanto è deciso il gesto".
            raw_conf = min(velocity / (min_v * 2), 1.0)
            timestamp_ms = int(frame_number / fps * 1000) if fps > 0 else 0

            self._last_strike_frame[fighter_id] = frame_number
            return StrikeEvent(
                fighter_id=fighter_id,
                strike_type=StrikeType(strike_name),
                timestamp_ms=timestamp_ms,
                frame_number=frame_number,
                confidence=round(raw_conf, 3),
                keypoint_id=kp_id,
            )

        return None

    def _passes_shape_check(self, strike_name: str, cfg: dict, h: deque, kp_id: int) -> bool:
        # Controlli aggiuntivi sulla forma del movimento.
        # Restituisce True se il movimento corrisponde al tipo di colpo.
        latest = h[-1]
        earliest = h[0]
        dx = latest[kp_id][0] - earliest[kp_id][0]
        dy = latest[kp_id][1] - earliest[kp_id][1]
        magnitude = math.hypot(dx, dy) or 1.0

        if strike_name == "uppercut":
            # Il vettore deve puntare principalmente verso l'alto (dy negativo in coords immagine).
            vertical_ratio = cfg.get("vertical_ratio", 0.6)
            return abs(dy) / magnitude > vertical_ratio and dy < 0

        if strike_name == "roundhouse_kick":
            # Il vettore deve avere una forte componente laterale.
            lateral_ratio = cfg.get("lateral_ratio", 0.5)
            return abs(dx) / magnitude > lateral_ratio

        if strike_name == "knee":
            # Il ginocchio deve salire (dy negativo).
            vertical_ratio = cfg.get("vertical_ratio", 0.7)
            return abs(dy) / magnitude > vertical_ratio and dy < 0

        if strike_name in ("hook",):
            # Il gomito deve essere piegato (angolo tra ~70° e ~110°).
            # Usiamo spalla-gomito-polso per il lato corrispondente.
            # Per semplicità usiamo i keypoint destri (5=spalla-dx, 8=gomito-dx, 10=polso-dx).
            shoulder = latest[6]   # spalla destra
            elbow = latest[8]      # gomito destro
            wrist = latest[10]     # polso destro
            if shoulder[2] < 0.3 or elbow[2] < 0.3 or wrist[2] < 0.3:
                return False
            angle = _angle_degrees(shoulder, elbow, wrist)
            return cfg.get("elbow_angle_min", 70) <= angle <= cfg.get("elbow_angle_max", 110)

        # Per jab, cross, front_kick, elbow: nessun controllo aggiuntivo oltre la velocità.
        return True
