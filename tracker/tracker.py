"""Multi-object tracker: Kalman prediction + Hungarian matching, SORT-style,
with ByteTrack's two-stage idea for low-confidence detections.

Every frame:
  1. predict   – each track's Kalman filter moves its box one frame forward
  2. stage 1   – match all live tracks to confident detections (score ≥ high_thresh)
  3. stage 2   – confirmed tracks still unmatched get a second chance with weak
                 detections (low_thresh ≤ score < high_thresh). From a drone, small
                 or partly hidden objects often get low scores; this keeps their ID.
  4. misses    – unmatched tracks age; tentative ones die, confirmed ones die after max_age
  5. births    – confident detections that no track explains start new tentative tracks
"""
from __future__ import annotations

import numpy as np

from .association import match
from .track import Track


class MultiObjectTracker:
    def __init__(self, high_thresh: float = 0.5, low_thresh: float = 0.1,
                 min_iou: float = 0.3, min_iou_low: float = 0.5,
                 min_hits: int = 3, max_age: int = 30):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.min_iou = min_iou
        self.min_iou_low = min_iou_low
        self.min_hits = min_hits
        self.max_age = max_age
        self.tracks: list[Track] = []
        self.frame_idx = 0
        self._next_id = 1

    def update(self, boxes, scores, classes) -> list[Track]:
        """Feed one frame of detections (xyxy boxes, scores, class ids).

        Returns the confirmed tracks that were matched in this frame.
        """
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        scores = np.asarray(scores, dtype=float).reshape(-1)
        classes = np.asarray(classes, dtype=int).reshape(-1)
        self.frame_idx += 1

        # 1. predict
        for t in self.tracks:
            t.predict()

        high = np.flatnonzero(scores >= self.high_thresh)
        low = np.flatnonzero((scores >= self.low_thresh) & (scores < self.high_thresh))

        # 2. stage 1: all live tracks vs. confident detections
        m1, unmatched_t, unmatched_high = match([t.box for t in self.tracks], boxes[high], self.min_iou)
        for ti, di in m1:
            d = high[di]
            self.tracks[ti].update(boxes[d], scores[d], classes[d])

        # 3. stage 2: unmatched confirmed tracks vs. weak detections
        cand = [ti for ti in unmatched_t if self.tracks[ti].is_confirmed]
        m2, _, _ = match([self.tracks[ti].box for ti in cand], boxes[low], self.min_iou_low)
        rescued = set()
        for ci, di in m2:
            d = low[di]
            self.tracks[cand[ci]].update(boxes[d], scores[d], classes[d])
            rescued.add(cand[ci])

        # 4. misses
        for ti in unmatched_t:
            if ti not in rescued:
                self.tracks[ti].mark_missed(self.max_age)

        # 5. births
        for di in unmatched_high:
            d = high[di]
            self.tracks.append(Track(self._next_id, boxes[d], scores[d], classes[d],
                                     self.frame_idx, self.min_hits))
            self._next_id += 1

        self.tracks = [t for t in self.tracks if not t.is_deleted]
        return [t for t in self.tracks if t.is_confirmed and t.time_since_update == 0]
