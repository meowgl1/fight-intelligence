from __future__ import annotations

from datetime import datetime, timezone

from events.schema import EventLog, StrikeEvent


class EventLogBuilder:
    # Accumula gli strike rilevati frame per frame e costruisce l'EventLog finale.
    # Va istanziato una volta per video, poi si chiama add_strike() a ogni colpo
    # rilevato, e alla fine build() per ottenere l'EventLog completo.

    def __init__(self, video_path: str, heuristic_version: str, classifier_model: str):
        self.video_path = video_path
        self.heuristic_version = heuristic_version
        self.classifier_model = classifier_model
        self._strikes: list[StrikeEvent] = []
        self._fighter_ids: set[int] = set()

    def add_strike(self, event: StrikeEvent) -> None:
        # Aggiunge uno strike alla lista e registra il fighter_id.
        # fighter_ids serve per calcolare fighter_count nell'EventLog finale.
        self._strikes.append(event)
        self._fighter_ids.add(event.fighter_id)

    def track_fighter(self, fighter_id: int) -> None:
        # Registra un fighter anche se non ha ancora tirato colpi.
        # Chiamato a ogni frame per i fighter tracciati — garantisce
        # che fighter_count rifletta quanti atleti sono stati visti nel video.
        self._fighter_ids.add(fighter_id)

    def build(self, video_fps: float, video_total_frames: int) -> EventLog:
        # Costruisce l'EventLog finale, ordinando gli strike per timestamp.
        # Il numero di fighter è limitato a 2 nell'MVP (vincolo data-model.md).
        strikes_sorted = sorted(self._strikes, key=lambda s: s.timestamp_ms)
        fighter_count = min(len(self._fighter_ids), 2)

        return EventLog(
            video_path=self.video_path,
            processed_at=datetime.now(tz=timezone.utc),
            heuristic_version=self.heuristic_version,
            classifier_model=self.classifier_model,
            video_fps=video_fps,
            video_total_frames=video_total_frames,
            fighter_count=fighter_count,
            strikes=strikes_sorted,
        )
