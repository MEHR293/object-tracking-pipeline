# Real-Time Object Detection & Multi-Object Tracking

An end-to-end pipeline that detects and tracks multiple objects simultaneously across live or
recorded video, then turns the tracks into **supply-chain style analytics**: items passing a
checkpoint, time spent in a staging area, and a per-object traceability log.

**Stack:** Python, YOLOv8 (Ultralytics), ByteTrack / BoT-SORT, OpenCV, NumPy

## How it maps to a supply chain

| Computer-vision concept | Logistics analogue |
|---|---|
| Persistent track ID | Asset / parcel identity across the whole scene |
| `LineCounter` (virtual line) | Dock-door or gate scan: items moving **in** vs **out** |
| `ZoneMonitor` (polygon) | Staging area: enter / exit events and **dwell time** |
| `events.csv` | Traceability log: who/what, when, where |
| Trails | Path an item took through the facility |

## Pipeline

```
video / webcam -> YOLO detection -> ByteTrack (persistent IDs) -> analytics -> annotated video + CSV + JSON
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# recorded video, count crossings of a horizontal line through the middle of the frame
python main.py --source path/to/video.mp4 --line 0,0.5,1,0.5

# live webcam with window, only people and cars
python main.py --source 0 --show --classes person,car

# dwell time in a region (polygon points as fractions of frame size)
python main.py --source path/to/video.mp4 --zone "0.6,0.2;0.95,0.2;0.95,0.9;0.6,0.9"
```

The YOLO weights (`yolov8n.pt`) download automatically on first run. Values in `--line` / `--zone`
that are all <= 1 are treated as fractions of the frame; otherwise as pixels.

### Outputs (in `outputs/`)
- `tracked.mp4`: boxes, IDs, trails, line/zone overlay, live counters
- `events.csv`: `frame, time_s, track_id, cls, event, detail` (`line_in`, `line_out`, `zone_enter`, `zone_exit`)
- `summary.json`: unique objects per class, line counts, dwell statistics, **measured processing FPS**

### Options
`--weights` (e.g. `yolov8s.pt`, or your own fine-tuned model) · `--tracker bytetrack.yaml|botsort.yaml` ·
`--conf` · `--iou` · `--classes` · `--device cpu|mps|0` · `--max-frames`

## Tests

```bash
pytest
```
Unit tests cover line crossing (direction, jitter, segment bounds), zone enter/exit/dwell
(including lost tracks), summaries and CLI parsing. They use synthetic tracks and don't need a GPU or weights.

## Known limitations (read before quoting results)
- **Pretrained COCO classes only** (person, car, truck, suitcase, ...). Detecting pallets, cartons or
  parcels requires fine-tuning on your own labelled data; pass it with `--weights`.
- Trackers can swap or drop IDs under heavy occlusion or low frame rates. Counts are only as good as the tracks.
- Speed depends heavily on hardware and model size; `summary.json` records the FPS measured on *your* machine.
- No accuracy benchmark (e.g. MOTA/IDF1) is included yet. Run on a labelled MOT clip to add one.

## License note
Ultralytics YOLO is licensed under AGPL-3.0 (commercial licenses available). Keep that in mind
if you build on this for a product.
