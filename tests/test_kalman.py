import numpy as np

from tracker.kalman import KalmanBoxFilter, cxcywh_to_xyxy, xyxy_to_cxcywh


def box_at(cx, cy, w=40, h=30):
    return cxcywh_to_xyxy([cx, cy, w, h])


def test_box_conversion_round_trip():
    b = np.array([10.0, 20.0, 50.0, 80.0])
    assert np.allclose(cxcywh_to_xyxy(xyxy_to_cxcywh(b)), b)


def test_learns_velocity_from_noisy_boxes():
    rng = np.random.default_rng(0)
    vx, vy = 4.0, -2.0
    kf = KalmanBoxFilter(box_at(100, 200))
    for k in range(1, 40):
        kf.predict()
        noisy = box_at(100 + vx * k, 200 + vy * k) + rng.normal(0, 1.0, 4)
        kf.update(noisy)
    est_vx, est_vy = kf.velocity
    assert abs(est_vx - vx) < 0.5 and abs(est_vy - vy) < 0.5


def test_prediction_coasts_through_missed_frames():
    kf = KalmanBoxFilter(box_at(0, 0))
    for k in range(1, 30):
        kf.predict()
        kf.update(box_at(5 * k, 0))
    # object hidden for 5 frames: predict only, no update
    for _ in range(5):
        kf.predict()
    cx = (kf.box[0] + kf.box[2]) / 2
    assert abs(cx - 5 * 34) < 3.0


def test_uncertainty_grows_on_predict_and_shrinks_on_update():
    kf = KalmanBoxFilter(box_at(50, 50))
    before = np.trace(kf.P)
    kf.predict()
    after_predict = np.trace(kf.P)
    kf.update(box_at(50, 50))
    assert after_predict > before
    assert np.trace(kf.P) < after_predict
