"""Drawing utilities for annotated output frames."""
from typing import Sequence

import cv2
import numpy as np

from objtrack.analytics import Analytics, Track


def color_for(track_id: int):
    """Stable, distinct BGR color per track id."""
    rng = np.random.default_rng(track_id * 9973 + 7)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


def draw_frame(frame, tracks: Sequence[Track], analytics: Analytics,
               line_px=None, zone_px=None, fps: float = 0.0):
    if zone_px is not None:
        pts = np.asarray(zone_px, dtype=np.int32).reshape(-1, 1, 2)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 200, 255))
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.polylines(frame, [pts], True, (0, 200, 255), 2)
    if line_px is not None:
        (x1, y1), (x2, y2) = line_px
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)

    for tr in tracks:
        col = color_for(tr.id)
        x1, y1, x2, y2 = (int(v) for v in tr.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        label = f"#{tr.id} {tr.cls}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        trail = analytics.trails.get(tr.id)
        if trail and len(trail) > 1:
            cv2.polylines(frame, [np.asarray(trail, dtype=np.int32).reshape(-1, 1, 2)], False, col, 2)

    # HUD
    lines = [f"FPS: {fps:.1f}", f"Active: {len(tracks)}  Unique: {sum(len(v) for v in analytics.unique.values())}"]
    if analytics.line:
        tin = sum(v["in"] for v in analytics.line.counts.values())
        tout = sum(v["out"] for v in analytics.line.counts.values())
        lines.append(f"Line  in: {tin}  out: {tout}")
    if analytics.zone:
        lines.append(f"In zone: {len(analytics.zone.inside)}")
    for i, text in enumerate(lines):
        y = 24 + i * 22
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame
