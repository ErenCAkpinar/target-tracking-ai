"""Command-line entry point.

Examples:
  python run_tracker.py --source data/drone.mp4
  python run_tracker.py --source 0 --max-frames 300          # webcam
  python run_tracker.py --source data/drone.mp4 --mpp 0.05   # 5 cm per pixel → km/h
"""
import argparse
import json

from tracker import MultiObjectTracker
from tracker.detector import DEFAULT_CLASSES, YoloDetector
from tracker.pipeline import run_to_files


def main() -> None:
    p = argparse.ArgumentParser(description="YOLOv8 + Kalman/Hungarian multi-object tracking")
    p.add_argument("--source", required=True, help="video file or camera index")
    p.add_argument("--out", default="outputs/run", help="output folder")
    p.add_argument("--weights", default="yolo11n-obb.pt",
                   help="yolo11n-obb.pt (DOTA, aerial; default) or yolov8n.pt (COCO, ground-level)")
    p.add_argument("--classes", nargs="*", default=list(DEFAULT_CLASSES))
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--device", default=None, help="cpu, mps or cuda:0 (default: auto)")
    p.add_argument("--fps", type=float, default=None, help="override the video's frame rate")
    p.add_argument("--mpp", type=float, default=None, help="metres per pixel, for km/h")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--min-hits", type=int, default=3)
    p.add_argument("--max-age", type=int, default=30)
    a = p.parse_args()

    det = YoloDetector(a.weights, classes=a.classes, imgsz=a.imgsz, device=a.device)
    trk = MultiObjectTracker(min_hits=a.min_hits, max_age=a.max_age)
    summary = run_to_files(a.source, a.out, det, trk, det.names, fps=a.fps,
                           metres_per_pixel=a.mpp, max_frames=a.max_frames)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
