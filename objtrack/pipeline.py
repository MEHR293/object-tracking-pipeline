"""End-to-end pipeline: video/webcam -> YOLO detection -> ByteTrack/BoT-SORT -> analytics -> outputs."""
import csv
import json
import time
from pathlib import Path
from typing import Optional

import cv2

from objtrack.analytics import Analytics, Track
from objtrack.utils import class_ids, parse_line, parse_zone, to_pixels
from objtrack.visualize import draw_frame


def _probe_fps(src) -> float:
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    return fps if fps and fps > 1 else 30.0


def run(source: str, weights: str = "yolov8n.pt", tracker: str = "bytetrack.yaml",
        conf: float = 0.3, iou: float = 0.5, classes: Optional[str] = None,
        line: Optional[str] = None, zone: Optional[str] = None,
        output_dir: str = "outputs", show: bool = False,
        max_frames: Optional[int] = None, device: Optional[str] = None) -> dict:
    from ultralytics import YOLO  # imported lazily so helpers/tests work without it

    model = YOLO(weights)
    src = int(source) if str(source).isdigit() else source
    live = isinstance(src, int)
    video_fps = 30.0 if live else _probe_fps(src)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    line_norm = parse_line(line) if line else None
    zone_norm = parse_zone(zone) if zone else None

    analytics = None
    writer = None
    line_px = zone_px = None
    t0 = time.perf_counter()
    last_tick = t0
    smooth_fps = 0.0
    idx = -1
    t = 0.0

    stream = model.track(source=src, stream=True, persist=True, tracker=tracker, conf=conf,
                         iou=iou, classes=class_ids(model.names, classes), verbose=False, device=device)
    try:
        for idx, r in enumerate(stream):
            if max_frames is not None and idx >= max_frames:
                break
            frame = r.orig_img.copy()
            h, w = frame.shape[:2]
            if analytics is None:
                if line_norm:
                    line_px = tuple(to_pixels(line_norm, w, h))
                if zone_norm:
                    zone_px = to_pixels(zone_norm, w, h)
                analytics = Analytics(line=line_px, zone=zone_px)
                writer = cv2.VideoWriter(str(out_dir / "tracked.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (w, h))

            tracks = []
            b = r.boxes
            if b is not None and b.id is not None:
                boxes = b.xyxy.cpu().numpy()
                ids = b.id.int().cpu().tolist()
                cls = b.cls.int().cpu().tolist()
                tracks = [Track(i, model.names[c], tuple(float(v) for v in bx))
                          for bx, i, c in zip(boxes, ids, cls)]

            now = time.perf_counter()
            t = (now - t0) if live else idx / video_fps
            inst = 1.0 / max(now - last_tick, 1e-6)
            smooth_fps = inst if smooth_fps == 0 else 0.9 * smooth_fps + 0.1 * inst
            last_tick = now

            analytics.update(idx, t, tracks)
            writer.write(draw_frame(frame, tracks, analytics, line_px, zone_px, smooth_fps))
            if show:
                cv2.imshow("Tracking (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    if analytics is None:
        raise RuntimeError(f"No frames read from source '{source}'. Check the path / camera index.")

    analytics.finish(idx, t)
    elapsed = time.perf_counter() - t0
    summary = analytics.summary()
    summary["processing_fps"] = round(analytics.frames / elapsed, 2) if elapsed > 0 else 0.0
    summary["weights"] = weights
    summary["tracker"] = tracker

    with open(out_dir / "events.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["frame", "time_s", "track_id", "cls", "event", "detail"])
        wr.writeheader()
        wr.writerows(analytics.events_as_dicts())
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
