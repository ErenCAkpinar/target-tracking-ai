"""Video/camera → detect → track → speed/direction → annotated frames and a track log."""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from .detector import Detections
from .draw import draw_tracks
from .motion import estimate_motion
from .tracker import MultiObjectTracker

LOG_FIELDS = ["frame", "track_id", "class", "score", "x1", "y1", "x2", "y2",
              "speed_px_s", "speed_kmh", "heading_deg", "compass"]


@dataclass
class FrameResult:
    index: int
    frame: np.ndarray
    annotated: np.ndarray
    rows: list[dict] = field(default_factory=list)
    detect_ms: float = 0.0      # time spent in the detector for this frame
    track_ms: float = 0.0       # time spent in the tracker + motion estimate


def open_source(source) -> cv2.VideoCapture:
    """A file path, or a camera index such as 0."""
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video source: {source}")
    return cap


def iter_tracking(source, detector: Callable[[np.ndarray], Detections], tracker: MultiObjectTracker,
                  names: dict[int, str], fps: float | None = None,
                  metres_per_pixel: float | None = None, max_frames: int | None = None) -> Iterator[FrameResult]:
    cap = open_source(source)
    fps = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    try:
        while max_frames is None or idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            t0 = time.perf_counter()
            det = detector(frame)
            t1 = time.perf_counter()
            tracks = tracker.update(det.boxes, det.scores, det.classes)
            motions = [estimate_motion(*t.velocity, fps=fps, metres_per_pixel=metres_per_pixel) for t in tracks]
            t2 = time.perf_counter()
            rows = []
            for t, m in zip(tracks, motions):
                x1, y1, x2, y2 = (round(float(v), 1) for v in t.box)
                rows.append({
                    "frame": idx, "track_id": t.id, "class": names.get(t.cls, str(t.cls)),
                    "score": round(t.score, 3), "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "speed_px_s": round(m.speed_px_s, 1),
                    "speed_kmh": None if m.speed_kmh is None else round(m.speed_kmh, 1),
                    "heading_deg": None if m.heading_deg is None else round(m.heading_deg, 1),
                    "compass": m.compass,
                })
            annotated = draw_tracks(frame.copy(), tracks, motions, names, frame_idx=idx)
            yield FrameResult(idx, frame, annotated, rows, (t1 - t0) * 1000, (t2 - t1) * 1000)
    finally:
        cap.release()


def run_to_files(source, out_dir, detector, tracker: MultiObjectTracker, names: dict[int, str],
                 fps: float | None = None, metres_per_pixel: float | None = None,
                 max_frames: int | None = None, warmup: int = 10) -> dict:
    """Write annotated.mp4 and tracks.csv into out_dir; return a short summary.

    The first `warmup` frames are slow (model load, GPU kernel compilation), so
    `steady_fps` leaves them out; `processing_fps` is the plain end-to-end figure.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = open_source(source)
    src_fps = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    writer = None
    ids: set[int] = set()
    frames = 0
    t0 = time.perf_counter()
    t_warm = t0 if warmup <= 0 else None
    detect_ms, track_ms = [], []
    with open(out_dir / "tracks.csv", "w", newline="") as f:
        log = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        log.writeheader()
        for res in iter_tracking(source, detector, tracker, names, src_fps, metres_per_pixel, max_frames):
            if writer is None:
                h, w = res.annotated.shape[:2]
                writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
            writer.write(res.annotated)
            log.writerows(res.rows)
            ids.update(r["track_id"] for r in res.rows)
            frames += 1
            if frames == warmup:
                t_warm = time.perf_counter()
            elif frames > warmup:
                detect_ms.append(res.detect_ms)
                track_ms.append(res.track_ms)
    if writer is not None:
        writer.release()
    end = time.perf_counter()
    elapsed = end - t0
    steady = len(detect_ms)
    return {"frames": frames, "unique_tracks": len(ids), "seconds": round(elapsed, 2),
            "processing_fps": round(frames / elapsed, 1) if elapsed > 0 else None,
            "steady_fps": round(steady / (end - t_warm), 1) if steady and t_warm else None,
            "detector_ms_median": round(float(np.median(detect_ms)), 1) if steady else None,
            "tracker_ms_median": round(float(np.median(track_ms)), 2) if steady else None,
            "video": str(out_dir / "annotated.mp4"), "log": str(out_dir / "tracks.csv")}
