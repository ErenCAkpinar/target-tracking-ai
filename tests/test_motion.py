import pytest

from tracker.motion import estimate_motion


def test_speed_in_pixels_per_second():
    m = estimate_motion(5.0, 0.0, fps=30)
    assert m.speed_px_s == pytest.approx(150.0)
    assert m.heading_deg == pytest.approx(0.0) and m.compass == "E"


def test_up_in_the_image_is_north():
    m = estimate_motion(0.0, -5.0, fps=30)      # y decreases = moving up
    assert m.heading_deg == pytest.approx(90.0) and m.compass == "N"


def test_diagonal_down_left_is_south_west():
    m = estimate_motion(-3.0, 3.0, fps=25)
    assert m.heading_deg == pytest.approx(225.0) and m.compass == "SW"


def test_kmh_with_ground_sample_distance():
    # 10 px/frame at 25 fps = 250 px/s; at 0.1 m/px that is 25 m/s = 90 km/h
    m = estimate_motion(10.0, 0.0, fps=25, metres_per_pixel=0.1)
    assert m.speed_kmh == pytest.approx(90.0)


def test_still_object_has_no_heading():
    m = estimate_motion(0.02, 0.01, fps=30)
    assert m.heading_deg is None and m.compass is None
