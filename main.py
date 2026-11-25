"""CLI entry point.

Examples:
  python main.py --source videos/warehouse.mp4 --line 0,0.5,1,0.5
  python main.py --source 0 --show --classes person,car
  python main.py --source videos/dock.mp4 --zone "0.6,0.2;0.95,0.2;0.95,0.9;0.6,0.9"
"""
import argparse
import json

from objtrack.pipeline import run


def main():
    p = argparse.ArgumentParser(description="Real-time object detection & multi-object tracking")
    p.add_argument("--source", required=True, help="video file path, or camera index like 0")
    p.add_argument("--weights", default="yolov8n.pt", help="YOLO weights (auto-downloaded if missing)")
    p.add_argument("--tracker", default="bytetrack.yaml", choices=["bytetrack.yaml", "botsort.yaml"])
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--classes", default=None, help="comma-separated names, e.g. person,car,truck")
    p.add_argument("--line", default=None, help="'x1,y1,x2,y2'; values <=1 are fractions of frame size")
    p.add_argument("--zone", default=None, help="'x,y;x,y;x,y;...' polygon; values <=1 are fractions")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--show", action="store_true", help="display a live window")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--device", default=None, help="e.g. cpu, mps, 0")
    a = p.parse_args()

    summary = run(a.source, a.weights, a.tracker, a.conf, a.iou, a.classes, a.line, a.zone,
                  a.output_dir, a.show, a.max_frames, a.device)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {a.output_dir}/tracked.mp4, events.csv, summary.json")


if __name__ == "__main__":
    main()
