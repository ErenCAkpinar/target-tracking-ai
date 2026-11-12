import numpy as np

from tracker import MultiObjectTracker
from tracker.kalman import cxcywh_to_xyxy


def box(cx, cy, w=30, h=30):
    return cxcywh_to_xyxy([cx, cy, w, h])


def ids_by_position(tracks):
    return {t.id: t.center for t in tracks}


def test_new_track_waits_for_min_hits():
    trk = MultiObjectTracker(min_hits=3)
    assert trk.update([box(100, 100)], [0.9], [2]) == []
    assert trk.update([box(102, 100)], [0.9], [2]) == []
    out = trk.update([box(104, 100)], [0.9], [2])
    assert len(out) == 1 and out[0].id == 1


def test_crossing_objects_keep_their_ids():
    """Two cars drive towards each other on nearby lanes and pass; IDs must not swap."""
    trk = MultiObjectTracker(min_hits=2)
    first_ids = {}
    for k in range(60):
        a = box(50 + 6 * k, 200)        # A drives right
        b = box(410 - 6 * k, 215)       # B drives left, 15 px lower (overlaps while passing)
        out = trk.update([a, b], [0.9, 0.9], [2, 2])
        for t in out:
            moving_right = t.velocity[0] > 0
            first_ids.setdefault(moving_right, t.id)
            assert first_ids[moving_right] == t.id, f"ID swap at frame {k}"
    assert len(first_ids) == 2


def test_track_survives_short_occlusion():
    trk = MultiObjectTracker(min_hits=2, max_age=10)
    for k in range(20):
        trk.update([box(100 + 5 * k, 100)], [0.9], [2])
    tid = trk.tracks[0].id
    for k in range(20, 26):                          # hidden for 6 frames (e.g. under a bridge)
        trk.update(np.zeros((0, 4)), [], [])
    out = trk.update([box(100 + 5 * 26, 100)], [0.9], [2])
    assert [t.id for t in out] == [tid]


def test_track_deleted_after_max_age():
    trk = MultiObjectTracker(min_hits=1, max_age=3)
    trk.update([box(100, 100)], [0.9], [0])
    for _ in range(4):
        trk.update(np.zeros((0, 4)), [], [])
    assert trk.tracks == []


def test_low_score_detection_keeps_confirmed_track_alive():
    """ByteTrack stage 2: a confirmed track can be matched to a weak (0.2) detection."""
    trk = MultiObjectTracker(min_hits=2, max_age=0)   # max_age 0: any miss would delete it
    for k in range(5):
        trk.update([box(100 + 4 * k, 100)], [0.9], [2])
    tid = trk.tracks[0].id
    out = trk.update([box(120, 100)], [0.2], [2])
    assert [t.id for t in out] == [tid]


def test_low_score_detection_does_not_start_a_track():
    trk = MultiObjectTracker(min_hits=1)
    assert trk.update([box(100, 100)], [0.3], [2]) == []
    assert trk.tracks == []


def test_class_is_majority_vote():
    trk = MultiObjectTracker(min_hits=1)
    out = []
    for k, cls in enumerate([2, 2, 7, 2]):           # one frame mislabels the car as a truck
        out = trk.update([box(100 + k, 100)], [0.9], [cls])
    assert out[0].cls == 2
