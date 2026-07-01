from __future__ import annotations

from ultralytics import YOLO


class PoseModel:
    # Carica il modello YOLO11-pose una sola volta e lo tiene in memoria.
    # Usare model.track() invece di model.predict() perché il tracking
    # assegna un ID stabile a ogni persona tra un frame e l'altro.

    def __init__(self, model_name: str = "yolo11n-pose.pt", tracker_config: str = "bytetrack.yaml"):
        # "n" sta per nano — il modello più leggero, adatto a CPU/MPS.
        # Il file si scarica automaticamente da ultralytics al primo avvio.
        self.model = YOLO(model_name)
        self.tracker_config = tracker_config
        self._last_result = None

    def annotated_frame(self):
        # Ritorna il frame con scheletro, bbox e ID sovrapposti da ultralytics.
        # Ritorna None se non è stato ancora processato nessun frame.
        return self._last_result.plot() if self._last_result is not None else None

    def process_frame(self, frame) -> list[dict]:
        # Ritorna una lista di fighter rilevati nel frame.
        # persist=True mantiene lo stato del tracker tra un frame e l'altro —
        # senza questa opzione gli ID cambiano a ogni chiamata.
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            verbose=False,
        )
        self._last_result = results[0] if results else None

        fighters = []
        if not results or results[0].boxes is None:
            return fighters

        r = results[0]
        boxes = r.boxes
        kps = r.keypoints

        # Se ByteTrack non ha ancora assegnato ID (es. primo frame),
        # boxes.id è None — saltiamo e aspettiamo il frame successivo.
        if boxes.id is None:
            return fighters

        n = len(boxes.id)
        for i in range(n):
            fighter_id = int(boxes.id[i].item())
            bbox = boxes.xyxy[i].cpu().numpy().tolist()  # [x1, y1, x2, y2] in pixel

            # keypoints: tensore [17, 3] con (x, y, confidence) per ogni punto COCO.
            if kps is not None:
                kp_data = kps.data[i].cpu().numpy().tolist()  # [[x,y,conf], ...]
            else:
                kp_data = [[0.0, 0.0, 0.0]] * 17

            fighters.append({
                "fighter_id": fighter_id,
                "bbox": bbox,
                "keypoints": kp_data,  # lista di 17 elementi [x, y, conf]
            })

        return fighters
