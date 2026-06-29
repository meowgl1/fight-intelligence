"""
Analisi video per sport da combattimento — entry point CLI.

Uso base:
    python main.py <video_path>

Con opzioni:
    python main.py <video_path> [--config <yaml_path>] [--out <output_path>]

Exit codes:
    0 = elaborazione completata con successo
    1 = file video non trovato
    2 = file config YAML non trovato o malformato
    3 = errore durante l'elaborazione
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import cv2

from classification.heuristics import HeuristicClassifier, load_config
from events.builder import EventLogBuilder
from vision.tracker import Tracker

# Config di default se l'utente non specifica --config
DEFAULT_CONFIG = Path(__file__).parent / "config" / "heuristics_v1.yaml"
CLASSIFIER_MODEL = "yolo11n-pose"


def parse_args() -> tuple[str, str, str]:
    # Parsing manuale degli argomenti senza argparse — sufficiente per l'MVP.
    # Formato: main.py <video> [--config <yaml>] [--out <json>]
    args = sys.argv[1:]
    if not args:
        print("Uso: python main.py <video_path> [--config <yaml>] [--out <json>]")
        sys.exit(1)

    video_path = args[0]
    config_path = str(DEFAULT_CONFIG)
    out_path = ""

    i = 1
    while i < len(args):
        if args[i] == "--config" and i + 1 < len(args):
            config_path = args[i + 1]
            i += 2
        elif args[i] == "--out" and i + 1 < len(args):
            out_path = args[i + 1]
            i += 2
        else:
            i += 1

    # Se l'output non è specificato, lo mettiamo accanto al video con suffisso _events.json
    if not out_path:
        stem = Path(video_path).stem
        out_path = f"{stem}_events.json"

    return video_path, config_path, out_path


def resolve_output_path(out_path: str) -> str:
    # Non sovrascriviamo file esistenti: aggiungiamo un suffisso numerico.
    # Esempio: se esiste run1.json, il prossimo diventa run1_1.json.
    p = Path(out_path)
    if not p.exists():
        return out_path
    stem = p.stem
    suffix = p.suffix
    parent = p.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return str(candidate)
        i += 1


def main():
    video_path, config_path, out_path = parse_args()

    # Verifica che il video esista prima di caricare i modelli (evita attese inutili).
    if not Path(video_path).exists():
        print(f"ERRORE: video non trovato: {video_path}", file=sys.stderr)
        sys.exit(1)

    # Carica la configurazione YAML con le soglie euristiche.
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        print(f"ERRORE: config non trovata — {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"ERRORE: config YAML malformata — {e}", file=sys.stderr)
        sys.exit(2)

    heuristic_version = config.get("version", "unknown")
    bytetrack_cfg = config.get("bytetrack", {})

    try:
        # Inizializziamo il tracker con i parametri ByteTrack dal config YAML.
        # Questo scrive un file YAML temporaneo e carica il modello YOLO.
        print(f"Carico il modello {CLASSIFIER_MODEL}...")
        tracker = Tracker(bytetrack_config=bytetrack_cfg, model_name=f"{CLASSIFIER_MODEL}.pt")
        classifier = HeuristicClassifier(config)
        builder = EventLogBuilder(
            video_path=video_path,
            heuristic_version=heuristic_version,
            classifier_model=CLASSIFIER_MODEL,
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"ERRORE: impossibile aprire il video: {video_path}", file=sys.stderr)
            sys.exit(1)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_number = 0

        print(f"Elaboro {video_path} — {total_frames} frame a {fps:.1f} fps")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Aggiorna il progress ogni 100 frame per non intasare il terminale.
            if frame_number % 100 == 0:
                pct = frame_number / total_frames * 100 if total_frames > 0 else 0
                print(f"Processing frame {frame_number}/{total_frames} ({pct:.1f}%)...")

            # Otteniamo i fighter tracciati in questo frame con i loro keypoint.
            fighters = tracker.get_fighters(frame)

            for f in fighters:
                builder.track_fighter(f["fighter_id"])
                event = classifier.update(
                    fighter_id=f["fighter_id"],
                    keypoints=f["keypoints"],
                    bbox=f["bbox"],
                    frame_number=frame_number,
                    fps=fps,
                )
                if event:
                    builder.add_strike(event)
                    print(f"  → Strike: {event.strike_type.value} (fighter {event.fighter_id}, frame {frame_number})")

            frame_number += 1

        cap.release()
        tracker.cleanup()

        # Costruiamo e serializziamo l'EventLog.
        log = builder.build(video_fps=fps, video_total_frames=frame_number)
        final_out = resolve_output_path(out_path)

        # model_dump() di pydantic serializza correttamente enum e datetime.
        with open(final_out, "w") as f_out:
            json.dump(log.model_dump(mode="json"), f_out, indent=2)

        print(f"\nDone. {len(log.strikes)} strike rilevati. Fighter tracciati: {log.fighter_count}.")
        print(f"Output: {final_out}")

    except KeyboardInterrupt:
        print("\nInterrotto dall'utente.")
        sys.exit(0)
    except Exception:
        print("ERRORE durante l'elaborazione:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()
