"""Aerial multi-object tracking: YOLOv8 detection + a from-scratch Kalman/Hungarian tracker."""
from .association import assign, iou_matrix, match
from .kalman import KalmanBoxFilter
from .motion import Motion, estimate_motion
from .track import Track, TrackState
from .tracker import MultiObjectTracker

__all__ = ["KalmanBoxFilter", "MultiObjectTracker", "Track", "TrackState", "Motion",
           "estimate_motion", "assign", "iou_matrix", "match"]
