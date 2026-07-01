# Fight Intelligence

Turn sparring footage into structured strike data. No cloud, no GPU required.

Fight Intelligence analyzes combat sports video (boxing, Muay Thai) using pose estimation and multi-object tracking to detect, classify, and timestamp every strike. It writes a versioned JSON event log you can query or feed into downstream analysis.

```bash
python main.py sparring.mp4
# → Done. 47 strikes detected. Fighters tracked: 2.
# → Output: sparring_events.json
```

---

## The problem

Reviewing sparring footage is slow. Coaches scrub through video manually to count jabs, spot patterns, and track fighter load. There is no cheap, local tool that does this automatically.

Fight Intelligence runs YOLO11-pose and ByteTrack on your video, classifies each strike from wrist and elbow velocity, and writes a JSON log you can actually use.

| Feature | Notes |
|---|---|
| 8 strike types | jab, cross, hook, uppercut, front kick, roundhouse, knee, elbow |
| Stable fighter IDs | IDs survive clinch and occlusion without swapping mid-round |
| Glove-color ReID | Disambiguates fighters even when bounding boxes overlap |
| YAML-driven thresholds | Tune detection without touching code |
| Versioned output | Every JSON records which config produced it |
| CPU/MPS only | Runs on a MacBook. No CUDA needed. |

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/your-username/fight-intelligence
cd fight-intelligence
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e .

# 2. Run on your video (model downloads on first run, ~6MB)
python main.py sparring.mp4

# 3. Inspect the output
python -c "
import json
log = json.load(open('sparring_events.json'))
print(f'{len(log[\"strikes\"])} strikes, {log[\"fighter_count\"]} fighters')
print({s[\"strike_type\"] for s in log[\"strikes\"]})
"
```

With options:

```bash
# Custom config and output path
python main.py sparring.mp4 --config config/heuristics_v1.yaml --out results/session1.json

# Live preview window (press Q to quit)
python main.py sparring.mp4 --preview
```

---

## How it works

```
Video frames
    │
    ▼
YOLO11-pose          (17 COCO keypoints per person, per frame)
    │
    ▼
ByteTrack            (stable fighter_id across frames)
    │
    ▼
Glove ReID           (color histogram to lock IDs through clinch)
    │
    ▼
Heuristic classifier (wrist/elbow velocity + direction -> strike_type)
    │
    ▼
EventLog (JSON)      (versioned, schema-validated output)
```

The classifier uses velocity rules, not a trained model. No labeled data needed. Thresholds live in `config/heuristics_v1.yaml` and are versioned alongside each output file.

---

## Output format

```json
{
  "video_path": "sparring.mp4",
  "processed_at": "2026-07-01T18:39:00Z",
  "heuristic_version": "1.0.0",
  "classifier_model": "yolo11n-pose",
  "fighter_count": 2,
  "total_frames": 540,
  "video_fps": 30.0,
  "strikes": [
    {
      "fighter_id": 1,
      "strike_type": "jab",
      "timestamp_ms": 66,
      "frame_number": 2,
      "confidence": 0.82,
      "keypoint_id": 10
    }
  ]
}
```

Full schema: [`specs/contracts/event-log-schema.json`](specs/001-fight-intelligence-mvp/contracts/event-log-schema.json)

---

## Configuration

All detection thresholds live in `config/heuristics_v1.yaml`. Change them without touching code; the output log records which version was used.

```yaml
version: "1.0.0"

jab:
  min_velocity: 0.15      # wrist speed threshold (normalized px/frame)
  direction_tolerance: 30 # degrees from forward axis

bytetrack:
  track_thresh: 0.5
  match_thresh: 0.8
  track_buffer: 30        # frames to keep a lost track alive
```

---

## Requirements

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) (recommended) or pip
- macOS (Apple Silicon or Intel) or Linux. No GPU required.
- `ultralytics`, `opencv-python`, `pydantic` (see `pyproject.toml`)

---

## Limitations

This is an MVP built to validate the CV pipeline. It works well enough for session review; for production labeling, you will want ground truth validation first.

| Limitation | Detail |
|---|---|
| Heuristic classifier | Velocity + direction rules, not ML. False positives on fast footwork. |
| No ground truth eval | Thresholds were hand-tuned on a small sample. Precision/recall unknown. |
| 8 strike types only | Spinning kicks, superman punches and similar are not detected. |
| Single camera | Multi-angle setups not supported. |
| Occlusion | Heavy clinch longer than 2 seconds can cause ID drift. |
| No audio | Round bell and corner calls are not used. |

---

## Roadmap

- [ ] Eval harness with labeled ground truth dataset
- [ ] Round segmentation (bell detection or manual timestamps)
- [ ] Per-fighter strike volume charts (exportable CSV)
- [ ] ONNX export for faster CPU inference
- [ ] Web UI for non-technical coaches

---

## License

MIT
