"""Supply-chain style analytics on top of multi-object tracks.

Mapping to logistics:
  track id            -> asset / parcel identity
  LineCounter         -> a gate or dock-door scan (items passing in / out)
  ZoneMonitor         -> a staging area (enter / exit events, dwell time)
  Analytics.events    -> a traceability log (exported to CSV)
"""
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


@dataclass
class Track:
    id: int
    cls: str
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2

    @property
    def centroid(self) -> Point:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class Event:
    frame: int
    time_s: float
    track_id: int
    cls: str
    event: str      # line_in | line_out | zone_enter | zone_exit
    detail: str = ""


class LineCounter:
    """Counts tracks crossing a line segment.

    Direction: moving from the negative side to the positive side of the
    directed line p1->p2 counts as 'in', the reverse as 'out'. A small margin
    around the line prevents jitter from double counting.
    """

    def __init__(self, p1: Point, p2: Point, margin: float = 2.0):
        self.p1 = np.asarray(p1, dtype=float)
        self.p2 = np.asarray(p2, dtype=float)
        self.d = self.p2 - self.p1
        self.length = float(np.hypot(*self.d))
        if self.length == 0:
            raise ValueError("Line endpoints must differ")
        self.margin = margin
        self.last_side: Dict[int, int] = {}
        self.counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"in": 0, "out": 0})

    def _signed_distance(self, pt: Point) -> float:
        v = np.asarray(pt, dtype=float) - self.p1
        return float((self.d[0] * v[1] - self.d[1] * v[0]) / self.length)

    def _within_segment(self, pt: Point) -> bool:
        v = np.asarray(pt, dtype=float) - self.p1
        t = float(np.dot(v, self.d) / (self.length ** 2))
        return 0.0 <= t <= 1.0

    def update(self, tr: Track, frame: int, t: float) -> Optional[Event]:
        dist = self._signed_distance(tr.centroid)
        if abs(dist) < self.margin:
            return None  # too close to the line to decide a side
        side = 1 if dist > 0 else -1
        prev = self.last_side.get(tr.id)
        self.last_side[tr.id] = side
        if prev is None or prev == side or not self._within_segment(tr.centroid):
            return None
        direction = "in" if side > 0 else "out"
        self.counts[tr.cls][direction] += 1
        return Event(frame, round(t, 3), tr.id, tr.cls, f"line_{direction}")


class ZoneMonitor:
    """Tracks entry, exit and dwell time inside a polygon."""

    def __init__(self, polygon: Sequence[Point], max_missing: int = 30):
        self.polygon = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
        self.max_missing = max_missing
        self.inside: Dict[int, dict] = {}
        self.dwell_times: List[float] = []

    def _contains(self, pt: Point) -> bool:
        return cv2.pointPolygonTest(self.polygon, (float(pt[0]), float(pt[1])), False) >= 0

    def _close(self, tid: int, frame: int, t: float, reason: str = "") -> Event:
        rec = self.inside.pop(tid)
        dwell = max(0.0, rec["last_t"] - rec["enter_t"])
        self.dwell_times.append(dwell)
        detail = f"dwell={dwell:.2f}s" + (f" ({reason})" if reason else "")
        return Event(frame, round(rec["last_t"], 3), tid, rec["cls"], "zone_exit", detail)

    def update(self, tracks: Sequence[Track], frame: int, t: float) -> List[Event]:
        events: List[Event] = []
        for tr in tracks:
            inside_now = self._contains(tr.centroid)
            if inside_now:
                if tr.id not in self.inside:
                    self.inside[tr.id] = {"enter_t": t, "last_t": t, "last_frame": frame, "cls": tr.cls}
                    events.append(Event(frame, round(t, 3), tr.id, tr.cls, "zone_enter"))
                else:
                    self.inside[tr.id].update(last_t=t, last_frame=frame)
            elif tr.id in self.inside:
                self.inside[tr.id]["last_t"] = t
                events.append(self._close(tr.id, frame, t))
        # tracks that vanished while inside the zone
        for tid in [k for k, v in self.inside.items() if frame - v["last_frame"] > self.max_missing]:
            events.append(self._close(tid, frame, t, reason="track lost"))
        return events

    def finish(self, frame: int, t: float) -> List[Event]:
        return [self._close(tid, frame, t, reason="end of video") for tid in list(self.inside)]


class Analytics:
    """Bundles trails, unique-object counts, line counting and zone monitoring."""

    def __init__(self, line: Optional[Tuple[Point, Point]] = None,
                 zone: Optional[Sequence[Point]] = None, trail_len: int = 30):
        self.line = LineCounter(*line) if line else None
        self.zone = ZoneMonitor(zone) if zone else None
        self.trails: Dict[int, Deque[Point]] = defaultdict(lambda: deque(maxlen=trail_len))
        self.last_seen: Dict[int, int] = {}
        self.unique: Dict[str, set] = defaultdict(set)
        self.events: List[Event] = []
        self.frames = 0

    def update(self, frame: int, t: float, tracks: Sequence[Track]) -> List[Event]:
        self.frames = frame + 1
        new: List[Event] = []
        for tr in tracks:
            self.trails[tr.id].append(tr.centroid)
            self.last_seen[tr.id] = frame
            self.unique[tr.cls].add(tr.id)
            if self.line:
                ev = self.line.update(tr, frame, t)
                if ev:
                    new.append(ev)
        if self.zone:
            new.extend(self.zone.update(tracks, frame, t))
        if frame % 200 == 0:  # drop trails of long-gone tracks
            for tid in [k for k, f in self.last_seen.items() if frame - f > 150]:
                self.trails.pop(tid, None)
                self.last_seen.pop(tid, None)
        self.events.extend(new)
        return new

    def finish(self, frame: int, t: float) -> List[Event]:
        new = self.zone.finish(frame, t) if self.zone else []
        self.events.extend(new)
        return new

    def summary(self) -> dict:
        out = {
            "frames": self.frames,
            "unique_objects": {c: len(ids) for c, ids in self.unique.items()},
            "total_unique_objects": sum(len(ids) for ids in self.unique.values()),
        }
        if self.line:
            out["line_counts"] = {c: dict(v) for c, v in self.line.counts.items()}
        if self.zone:
            d = self.zone.dwell_times
            out["zone"] = {
                "visits": len(d),
                "mean_dwell_s": round(float(np.mean(d)), 3) if d else 0.0,
                "max_dwell_s": round(float(np.max(d)), 3) if d else 0.0,
            }
        return out

    def events_as_dicts(self) -> List[dict]:
        return [asdict(e) for e in self.events]
