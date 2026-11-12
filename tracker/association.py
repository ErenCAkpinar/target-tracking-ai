"""Matching predicted track boxes to new detections.

Cost: IoU (intersection over union) between each predicted box and each detection.
Solver: the Hungarian algorithm (scipy's linear_sum_assignment), which finds the
pairing with the highest total IoU. Greedy "best pair first" matching can get
this wrong when two objects are close (see tests/test_association.py).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a, b) -> np.ndarray:
    """IoU between every box in a (N×4, xyxy) and every box in b (M×4). Returns N×M."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def assign(iou: np.ndarray, min_iou: float):
    """Hungarian assignment on an IoU matrix, then drop pairs below min_iou (gating).

    Returns (matches, unmatched_rows, unmatched_cols); matches is a list of (row, col).
    """
    n_rows, n_cols = iou.shape
    if iou.size == 0:
        return [], list(range(n_rows)), list(range(n_cols))
    rows, cols = linear_sum_assignment(-iou)          # maximise total IoU
    matches = [(int(r), int(c)) for r, c in zip(rows, cols) if iou[r, c] >= min_iou]
    used_r = {r for r, _ in matches}
    used_c = {c for _, c in matches}
    return (matches,
            [r for r in range(n_rows) if r not in used_r],
            [c for c in range(n_cols) if c not in used_c])


def match(track_boxes, det_boxes, min_iou: float):
    return assign(iou_matrix(track_boxes, det_boxes), min_iou)
