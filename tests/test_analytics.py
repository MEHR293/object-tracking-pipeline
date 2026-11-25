from objtrack.analytics import Analytics, LineCounter, Track, ZoneMonitor
from objtrack.utils import class_ids, parse_line, parse_zone, to_pixels
import pytest


def box_at(cx, cy, s=10):
    return (cx - s, cy - s, cx + s, cy + s)


def tr(i, cx, cy, cls="person"):
    return Track(i, cls, box_at(cx, cy))


def test_line_counts_in_and_out():
    lc = LineCounter((0, 100), (200, 100))  # horizontal line at y=100
    evs = []
    for f, y in enumerate([60, 80, 120, 140]):          # moving down: negative->positive side
        evs.append(lc.update(tr(1, 100, y), f, f / 10))
    assert [e.event for e in evs if e] == ["line_in"]    # one crossing, reported once
    assert lc.counts["person"] == {"in": 1, "out": 0}   # y increasing is the positive side here
    for f, y in enumerate([140, 120, 80, 60], start=4):  # back up
        lc.update(tr(1, 100, y), f, f / 10)
    assert lc.counts["person"] == {"in": 1, "out": 1}


def test_line_ignores_crossing_outside_segment():
    lc = LineCounter((0, 100), (50, 100))
    lc.update(tr(1, 150, 60), 0, 0)
    assert lc.update(tr(1, 150, 140), 1, 0.1) is None
    assert lc.counts == {}


def test_line_jitter_inside_margin_not_counted():
    lc = LineCounter((0, 100), (200, 100), margin=5)
    for f, y in enumerate([90, 101, 99, 102, 98, 90]):
        lc.update(tr(1, 100, y), f, f / 10)
    assert not lc.counts


def test_zone_enter_exit_and_dwell():
    zm = ZoneMonitor([(100, 100), (200, 100), (200, 200), (100, 200)])
    events = []
    for f, x in enumerate([50, 150, 160, 170, 250]):    # outside, in, in, in, outside
        events += zm.update([tr(1, x, 150)], f, f * 1.0)
    kinds = [e.event for e in events]
    assert kinds == ["zone_enter", "zone_exit"]
    assert zm.dwell_times == [pytest.approx(3.0)]       # inside from t=1 until the exit frame at t=4


def test_zone_closes_lost_track():
    zm = ZoneMonitor([(0, 0), (100, 0), (100, 100), (0, 100)], max_missing=5)
    zm.update([tr(7, 50, 50)], 0, 0.0)
    out = zm.update([], 10, 1.0)  # track vanished for > max_missing frames
    assert [e.event for e in out] == ["zone_exit"]
    assert "track lost" in out[0].detail


def test_zone_finish_flushes_open_visits():
    zm = ZoneMonitor([(0, 0), (100, 0), (100, 100), (0, 100)])
    zm.update([tr(3, 50, 50)], 0, 0.0)
    out = zm.finish(5, 0.5)
    assert len(out) == 1 and "end of video" in out[0].detail


def test_analytics_summary_and_unique_counts():
    a = Analytics(line=((0, 100), (200, 100)), zone=[(0, 0), (200, 0), (200, 90), (0, 90)])
    for f, y in enumerate([50, 70, 130]):
        a.update(f, f * 0.1, [tr(1, 100, y, "box"), tr(2, 100, y, "person")])
    a.finish(2, 0.2)
    s = a.summary()
    assert s["unique_objects"] == {"box": 1, "person": 1}
    assert s["total_unique_objects"] == 2
    assert s["line_counts"]["box"]["in"] == 1
    assert s["zone"]["visits"] == 2


def test_parsers():
    assert parse_line("0,0.5,1,0.5") == ((0, 0.5), (1, 0.5))
    assert len(parse_zone("0,0;1,0;1,1")) == 3
    with pytest.raises(ValueError):
        parse_line("1,2,3")
    with pytest.raises(ValueError):
        parse_zone("0,0;1,1")
    assert to_pixels([(0.5, 0.5)], 200, 100) == [(100.0, 50.0)]
    assert to_pixels([(150, 50)], 200, 100) == [(150.0, 50.0)]
    names = {0: "person", 2: "car"}
    assert class_ids(names, "person,car") == [0, 2]
    assert class_ids(names, None) is None
    with pytest.raises(ValueError):
        class_ids(names, "dragon")
