"""Small helpers: parsing CLI geometry and resolving class names."""
from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, float]


def parse_line(s: str) -> Tuple[Point, Point]:
    """'x1,y1,x2,y2' -> ((x1,y1),(x2,y2))."""
    vals = [float(v) for v in s.split(",")]
    if len(vals) != 4:
        raise ValueError("Line must be 'x1,y1,x2,y2'")
    return (vals[0], vals[1]), (vals[2], vals[3])


def parse_zone(s: str) -> List[Point]:
    """'x,y;x,y;x,y;...' (>=3 points) -> list of points."""
    pts = []
    for part in s.split(";"):
        x, y = part.split(",")
        pts.append((float(x), float(y)))
    if len(pts) < 3:
        raise ValueError("Zone needs at least 3 points: 'x,y;x,y;x,y'")
    return pts


def to_pixels(points: Sequence[Point], width: int, height: int) -> List[Point]:
    """If every value is <= 1.0 treat points as fractions of the frame size."""
    if all(0.0 <= v <= 1.0 for p in points for v in p):
        return [(x * width, y * height) for x, y in points]
    return [(float(x), float(y)) for x, y in points]


def class_ids(names: dict, wanted: Optional[str]) -> Optional[List[int]]:
    """'person,car' -> [0, 2] using the model's {id: name} map. None -> all classes."""
    if not wanted:
        return None
    lookup = {v.lower(): k for k, v in names.items()}
    ids = []
    for w in (x.strip().lower() for x in wanted.split(",") if x.strip()):
        if w not in lookup:
            raise ValueError(f"Unknown class '{w}'. Available: {sorted(lookup)}")
        ids.append(lookup[w])
    return ids
