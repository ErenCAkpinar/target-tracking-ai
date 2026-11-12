"""YOLO detector wrapper: frame in, (boxes, scores, classes) out.

Kept separate from the tracker, so the tracker can be tested without a model
and the detector can be swapped. Two kinds of Ultralytics weights work:

- ordinary detection models (e.g. COCO `yolov8n.pt`): axis-aligned boxes
- oriented-box (OBB) models (e.g. DOTA `yolo11n-obb.pt`, trained on aerial
  images): rotated boxes, which we turn into the axis-aligned box around them
  because the tracker's Kalman state is (cx, cy, w, h)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Classes worth tracking from the air: COCO names, then DOTA names, then VisDrone names.
# Only the names a given model knows are used.
DEFAULT_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck",
                   "small vehicle", "large vehicle",
                   "pedestrian", "people", "van", "tricycle", "awning-tricycle", "motor")


@dataclass
class Detections:
    boxes: np.ndarray     # N×4, xyxy pixels
    scores: np.ndarray    # N
    classes: np.ndarray   # N, integer class ids

    @staticmethod
    def empty() -> "Detections":
        return Detections(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int))


def pick_device() -> str:
    """Apple GPU (MPS) or CUDA when available; Ultralytics would otherwise default to CPU on a Mac."""
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class YoloDetector:
    def __init__(self, weights: str = "yolo11n-obb.pt", classes=DEFAULT_CLASSES, conf: float = 0.1,
                 imgsz: int = 960, min_box_area: float = 16.0, device: str | None = None):
        from ultralytics import YOLO        # imported here so tests don't need torch

        self.model = YOLO(weights)
        self.names: dict[int, str] = dict(self.model.names)
        wanted = set(classes) if classes else None
        self.class_ids = [i for i, n in self.names.items() if n in wanted] if wanted else None
        if self.class_ids == []:
            raise ValueError(f"None of {sorted(wanted or [])} is a class of {weights}; "
                             f"it knows: {sorted(self.names.values())}")
        # conf is low on purpose: the tracker's stage 2 uses 0.1–0.5 detections to keep IDs alive
        self.conf = conf
        self.imgsz = imgsz              # larger than the default 640 helps with small aerial objects
        self.min_box_area = min_box_area
        self.device = device or pick_device()

    def __call__(self, frame: np.ndarray) -> Detections:
        r = self.model.predict(frame, conf=self.conf, classes=self.class_ids, imgsz=self.imgsz,
                               device=self.device, verbose=False)[0]
        # OBB models fill r.obb instead of r.boxes; .xyxy is the axis-aligned box around the rotated one
        res = r.obb if getattr(r, "obb", None) is not None else r.boxes
        if res is None or len(res) == 0:
            return Detections.empty()
        boxes = res.xyxy.cpu().numpy()
        scores = res.conf.cpu().numpy()
        classes = res.cls.cpu().numpy().astype(int)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = area >= self.min_box_area
        return Detections(boxes[keep], scores[keep], classes[keep])
