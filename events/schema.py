from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class StrikeType(str, Enum):
    # I tipi di colpo che il sistema sa riconoscere nell'MVP.
    # Sono una lista chiusa: nessun "altro" o "sconosciuto" per ora.
    JAB = "jab"
    CROSS = "cross"
    HOOK = "hook"
    UPPERCUT = "uppercut"
    FRONT_KICK = "front_kick"
    ROUNDHOUSE_KICK = "roundhouse_kick"
    KNEE = "knee"
    ELBOW = "elbow"


class StrikeEvent(BaseModel):
    # Un singolo colpo rilevato nel video.
    # fighter_id viene da ByteTrack — identifica CHI ha tirato il colpo.
    fighter_id: int = Field(ge=1)
    strike_type: StrikeType
    timestamp_ms: int = Field(ge=0)
    frame_number: int = Field(ge=0)
    # confidence va da 0 a 1: quanto è sicura l'euristica di questa classificazione.
    confidence: float = Field(ge=0.0, le=1.0)
    # keypoint_id è l'ID COCO del punto chiave che ha triggherato il rilevamento
    # (es. 10 = polso destro per un jab destro).
    keypoint_id: int = Field(ge=0, le=16)


class EventLog(BaseModel):
    # L'output completo dell'analisi di un video.
    # heuristic_version e classifier_model servono a sapere QUALI regole erano
    # attive durante questa run — indispensabile per confrontare run successive.
    video_path: str
    processed_at: datetime
    heuristic_version: str
    classifier_model: str
    video_fps: float = Field(gt=0)
    video_total_frames: int = Field(ge=0)
    fighter_count: int = Field(ge=0, le=2)
    strikes: list[StrikeEvent] = Field(default_factory=list)
