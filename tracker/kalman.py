"""Constant-velocity Kalman filter for a single bounding box.

State (8 numbers):       x = [cx, cy, w, h, vx, vy, vw, vh]
Measurement (4 numbers): z = [cx, cy, w, h]

cx, cy are the box centre in pixels, w and h its size, and v* how much each of
those changes per frame. The detector only sees the box; the filter infers the
velocity from how the box moves over time.

Noise is scaled by box size (the DeepSORT convention): a 200-px truck jitters
by more pixels than a 15-px car seen from a drone.
"""
from __future__ import annotations

import numpy as np

# Standard deviations as a fraction of the box's longer side.
STD_POS = 1.0 / 20    # how far the measured box may be off, and position process noise
STD_VEL = 1.0 / 160   # how much the velocity may change between frames


def xyxy_to_cxcywh(box) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=float)


def cxcywh_to_xyxy(z) -> np.ndarray:
    cx, cy, w, h = z
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=float)


class KalmanBoxFilter:
    def __init__(self, box_xyxy, dt: float = 1.0):
        z = xyxy_to_cxcywh(box_xyxy)
        self.x = np.concatenate([z, np.zeros(4)])          # velocity starts unknown → 0

        # F (motion model): position_next = position + velocity * dt
        self.F = np.eye(8)
        self.F[:4, 4:] = dt * np.eye(4)

        # H (measurement model): the detector observes the first four state entries
        self.H = np.eye(4, 8)

        # P (state uncertainty): position roughly known, velocity very uncertain
        s = self._scale()
        std = [2 * STD_POS * s] * 4 + [10 * STD_VEL * s] * 4
        self.P = np.diag(np.square(std))

    def _scale(self) -> float:
        return max(float(self.x[2]), float(self.x[3]), 1.0)

    def predict(self) -> np.ndarray:
        """Move the state one frame forward. Uncertainty grows by Q."""
        s = self._scale()
        Q = np.diag(np.square([STD_POS * s] * 4 + [STD_VEL * s] * 4))
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q
        self.x[2:4] = np.maximum(self.x[2:4], 1.0)         # a box can't shrink below 1 px
        return self.box

    def update(self, box_xyxy) -> np.ndarray:
        """Correct the prediction with a detection. Uncertainty shrinks."""
        z = xyxy_to_cxcywh(box_xyxy)
        R = np.diag(np.square([STD_POS * self._scale()] * 4))   # detector noise
        y = z - self.H @ self.x                                  # innovation: measured − predicted
        S = self.H @ self.P @ self.H.T + R                       # innovation covariance
        K = np.linalg.solve(S, self.H @ self.P).T                # gain K = P Hᵀ S⁻¹ (P, S symmetric)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        return self.box

    @property
    def box(self) -> np.ndarray:
        return cxcywh_to_xyxy(self.x[:4])

    @property
    def velocity(self) -> tuple[float, float]:
        """Centre velocity in pixels per frame (vx right, vy down)."""
        return float(self.x[4]), float(self.x[5])
