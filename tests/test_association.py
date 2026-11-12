import numpy as np

from tracker.association import assign, iou_matrix, match


def test_iou_values():
    a = [[0, 0, 10, 10]]
    b = [[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]]
    iou = iou_matrix(a, b)
    assert np.allclose(iou, [[1.0, 50 / 150, 0.0]])


def test_empty_inputs():
    m, ut, ud = match([], [[0, 0, 1, 1]], 0.3)
    assert m == [] and ut == [] and ud == [0]
    m, ut, ud = match([[0, 0, 1, 1]], np.zeros((0, 4)), 0.3)
    assert m == [] and ut == [0] and ud == []


def test_hungarian_beats_greedy():
    # Greedy takes the single best pair (T0-D0 = 0.6) and leaves T1 with nothing.
    # Hungarian maximises the total: T0-D1 + T1-D0 = 1.0 > 0.6.
    iou = np.array([[0.6, 0.5],
                    [0.5, 0.0]])
    matches, ut, ud = assign(iou, min_iou=0.3)
    assert sorted(matches) == [(0, 1), (1, 0)]
    assert ut == [] and ud == []


def test_gating_rejects_weak_overlaps():
    iou = np.array([[0.2]])
    matches, ut, ud = assign(iou, min_iou=0.3)
    assert matches == [] and ut == [0] and ud == [0]
