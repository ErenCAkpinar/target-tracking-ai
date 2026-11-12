"""End-to-end run on a synthetic video, with a colour-blob 'detector' instead of YOLO.

Checks the whole path (video → detect → track → motion → annotated mp4 + CSV)
without needing model weights or a GPU.
"""
import csv

import cv2
import numpy as np
import pytest

from tracker import MultiObjectTracker
from tracker.detector import Detections
from tracker.pipeline import run_to_files

FPS = 20


def make_video(path, n_frames=40, size=(320, 240)):
    w, h = size
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (w, h))
    for k in range(n_frames):
        img = np.zeros((h, w, 3), np.uint8)
        x = 20 + 4 * k                     # square A: 4 px/frame to the right
        cv2.rectangle(img, (x, 60), (x + 24, 84), (0, 0, 255), -1)
        y = 200 - 3 * k                    # square B: 3 px/frame upwards
        cv2.rectangle(img, (250, y), (274, y + 24), (0, 255, 0), -1)
        vw.write(img)
    vw.release()


def blob_detector(frame):
    mask = (frame.max(axis=2) > 100).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = [[x, y, x + bw, y + bh] for x, y, bw, bh, area in stats[1:] if area > 50]
    if not boxes:
        return Detections.empty()
    return Detections(np.array(boxes, float), np.full(len(boxes), 0.9), np.full(len(boxes), 2))


def test_synthetic_video_end_to_end(tmp_path):
    src = tmp_path / "synthetic.avi"
    make_video(src)
    summary = run_to_files(src, tmp_path / "out", blob_detector, MultiObjectTracker(min_hits=3),
                           names={2: "car"})
    assert summary["frames"] == 40
    assert summary["unique_tracks"] == 2
    assert (tmp_path / "out" / "annotated.mp4").stat().st_size > 0

    rows = list(csv.DictReader(open(summary["log"])))
    last = {r["track_id"]: r for r in rows}            # last row per track
    speeds = sorted(float(r["speed_px_s"]) for r in last.values())
    assert speeds[0] == pytest.approx(3 * FPS, rel=0.1)   # B: 60 px/s
    assert speeds[1] == pytest.approx(4 * FPS, rel=0.1)   # A: 80 px/s
    assert {r["compass"] for r in last.values()} == {"E", "N"}
