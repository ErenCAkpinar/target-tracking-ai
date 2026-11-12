"""One tracked object: its Kalman filter, ID, class votes and life cycle.

Life cycle:
  TENTATIVE  → just born; must be matched min_hits frames in a row to be trusted
  CONFIRMED  → shown to the user; survives up to max_age frames without a match
  DELETED    → removed (a tentative track dies at its first miss)
"""
from __future__ import annotations

from collections import Counter
from enum import Enum

import numpy as np

from .kalman import KalmanBoxFilter

TRAIL_LEN = 30   # how many recent centres to keep for drawing the trail


class TrackState(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    DELETED = "deleted"


class Track:
    def __init__(self, track_id: int, box, score: float, cls: int, frame_idx: int, min_hits: int):
        self.id = track_id
        self.kf = KalmanBoxFilter(box)
        self.score = float(score)
        self.class_votes: Counter[int] = Counter({int(cls): 1})
        self.hits = 1
        self.time_since_update = 0
        self.min_hits = min_hits
        self.first_frame = frame_idx
        self.state = TrackState.CONFIRMED if min_hits <= 1 else TrackState.TENTATIVE
        self.trail: list[tuple[float, float]] = [self.center]

    # -- read-only views -------------------------------------------------
    @property
    def box(self) -> np.ndarray:
        return self.kf.box

    @property
    def center(self) -> tuple[float, float]:
        return float(self.kf.x[0]), float(self.kf.x[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return self.kf.velocity

    @property
    def cls(self) -> int:
        """Majority vote over all matched detections, so one wrong label doesn't flip the class."""
        return self.class_votes.most_common(1)[0][0]

    @property
    def is_confirmed(self) -> bool:
        return self.state is TrackState.CONFIRMED

    @property
    def is_deleted(self) -> bool:
        return self.state is TrackState.DELETED

    # -- per-frame steps -------------------------------------------------
    def predict(self) -> None:
        self.kf.predict()
        self.time_since_update += 1

    def update(self, box, score: float, cls: int) -> None:
        self.kf.update(box)
        self.score = float(score)
        self.class_votes[int(cls)] += 1
        self.hits += 1
        self.time_since_update = 0
        if self.state is TrackState.TENTATIVE and self.hits >= self.min_hits:
            self.state = TrackState.CONFIRMED
        self.trail.append(self.center)
        del self.trail[:-TRAIL_LEN]

    def mark_missed(self, max_age: int) -> None:
        if self.state is TrackState.TENTATIVE or self.time_since_update > max_age:
            self.state = TrackState.DELETED
