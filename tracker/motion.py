"""Speed and direction from a track's Kalman velocity.

The velocity is in pixels per frame, so speed in pixels per second is |v| × fps.
With a ground sample distance (metres per pixel, known from altitude and camera)
it becomes m/s and km/h.

Directions are in image space: 0° = right, 90° = up (towards the top of the
frame). They are not geographic north unless the camera points north. Speeds
are relative to the image; if the drone itself moves, its own motion is
included (see docs for ego-motion compensation).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

COMPASS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]   # 45° sectors, image-up = N


@dataclass(frozen=True)
class Motion:
    speed_px_s: float
    speed_kmh: float | None      # only when metres_per_pixel is given
    heading_deg: float | None    # None when the object is (almost) still
    compass: str | None


def estimate_motion(vx: float, vy: float, fps: float,
                    metres_per_pixel: float | None = None,
                    min_speed_px_s: float = 3.0) -> Motion:
    speed_px_s = math.hypot(vx, vy) * fps
    speed_kmh = speed_px_s * metres_per_pixel * 3.6 if metres_per_pixel else None
    if speed_px_s < min_speed_px_s:      # below this, the heading is just noise
        return Motion(speed_px_s, speed_kmh, None, None)
    heading = math.degrees(math.atan2(-vy, vx)) % 360   # image y grows downward → flip it
    compass = COMPASS[int((heading + 22.5) // 45) % 8]
    return Motion(speed_px_s, speed_kmh, heading, compass)
