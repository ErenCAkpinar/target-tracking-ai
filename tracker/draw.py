"""Drawing boxes, IDs, speed/direction labels and trails onto frames."""
from __future__ import annotations

import colorsys

import cv2
import numpy as np

from .motion import Motion
from .track import Track


def color_for(track_id: int) -> tuple[int, int, int]:
    """A stable, well-spread colour per ID (golden-ratio hue steps), as BGR."""
    h = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_tracks(frame: np.ndarray, tracks: list[Track], motions: list[Motion],
                names: dict[int, str], frame_idx: int | None = None) -> np.ndarray:
    for t, m in zip(tracks, motions):
        c = color_for(t.id)
        x1, y1, x2, y2 = (int(round(v)) for v in t.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        if len(t.trail) > 1:
            pts = np.array(t.trail, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], False, c, 2)
        label = f"#{t.id} {names.get(t.cls, t.cls)}"
        if m.compass:
            speed = f"{m.speed_kmh:.0f} km/h" if m.speed_kmh is not None else f"{m.speed_px_s:.0f} px/s"
            label += f" {speed} {m.compass}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), c, -1)
        cv2.putText(frame, label, (x1 + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    if frame_idx is not None:
        hud = f"frame {frame_idx}  tracks {len(tracks)}"
        (tw, th), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (0, 0), (tw + 20, th + 18), (0, 0, 0), -1)   # dark box so labels can't hide it
        cv2.putText(frame, hud, (10, th + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame
