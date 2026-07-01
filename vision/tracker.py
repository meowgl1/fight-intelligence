from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from vision.pose import PoseModel


class Tracker:
    # Wrapper attorno a PoseModel che gestisce la configurazione di ByteTrack.
    # Scrive un file YAML temporaneo con i parametri scelti dall'utente,
    # poi lo passa a YOLO — così possiamo cambiare le soglie senza toccare il codice.

    def __init__(self, bytetrack_config: dict, model_name: str = "yolo11n-pose.pt"):
        # Creiamo un file YAML temporaneo con i parametri ByteTrack.
        # Il file viene eliminato quando l'oggetto viene distrutto.
        self._tmp_config = self._write_tracker_config(bytetrack_config)
        self._pose_model = PoseModel(model_name=model_name, tracker_config=self._tmp_config)

    def _write_tracker_config(self, config: dict) -> str:
        # ByteTrack accetta un file YAML con queste chiavi.
        # track_buffer: quanti frame aspettare prima di perdere un fighter dall'occhio
        # — aumentarlo aiuta durante il clinch dove un fighter può sparire per qualche frame.
        tracker_yaml = {
            "tracker_type": "bytetrack",
            "track_high_thresh": config.get("track_thresh", 0.25),
            "track_low_thresh": 0.1,
            "new_track_thresh": config.get("track_thresh", 0.25),
            "track_buffer": config.get("track_buffer", 30),
            "match_thresh": config.get("match_thresh", 0.8),
            "fuse_score": False,
        }
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, prefix="bytetrack_"
        )
        yaml.dump(tracker_yaml, tmp)
        tmp.flush()
        return tmp.name

    def get_fighters(self, frame) -> list[dict]:
        # Restituisce la lista di fighter rilevati nel frame,
        # ciascuno con fighter_id stabile, bbox e 17 keypoint COCO.
        return self._pose_model.process_frame(frame)

    def annotated_frame(self):
        return self._pose_model.annotated_frame()

    def cleanup(self):
        # Elimina il file YAML temporaneo creato nel costruttore.
        Path(self._tmp_config).unlink(missing_ok=True)
