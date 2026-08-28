from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def create_sample_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (320, 240), (18, 28, 36))
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 50, 240, 190), fill=(220, 190, 80))
    draw.ellipse((120, 80, 200, 160), fill=(40, 145, 210))
    image.save(path)
    return path


def create_sample_video(path: Path, fps: int = 12, frames: int = 48) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),  # ty: ignore[unresolved-attribute]
        fps,
        (320, 240),
    )
    if not writer.isOpened():
        raise RuntimeError("Unable to create synthetic video")

    for index in range(frames):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:, :] = (18, 28, 36)
        if index < frames // 3:
            x = 50
        elif index < (frames * 2) // 3:
            x = 50 + (index - frames // 3) * 8
        else:
            x = 210
        cv2.rectangle(frame, (x, 90), (x + 50, 140), (70, 210, 160), -1)
        cv2.putText(frame, str(index), (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
        writer.write(frame)

    writer.release()
    return path


def create_live_frame(index: int) -> np.ndarray:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[:, :] = (12, 24, 48)
    cv2.circle(frame, (30 + index * 12, 60), 18, (220, 120, 70), -1)
    return frame


def _panning_texture(width: int, height: int, seed: int) -> np.ndarray:
    """A wide, high-frequency backdrop for the camera to pan across."""
    rng = np.random.default_rng(seed)
    texture = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (0, 0), sigmaX=2.0)
    for _ in range(60):
        top_left = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        size = int(rng.integers(10, 45))
        color = tuple(int(c) for c in rng.integers(30, 230, 3))
        cv2.rectangle(
            texture,
            top_left,
            (top_left[0] + size, top_left[1] + size),
            color,
            -1,
        )
    return texture


def create_motion_video(
    path: Path,
    *,
    fps: int = 24,
    seconds: int = 20,
    motion_windows: tuple[tuple[float, float], ...] = ((5.0, 9.0), (13.0, 16.0)),
    speed_px_per_frame: int = 6,
    camera_pan: bool = True,
    seed: int = 0,
) -> list[tuple[float, float]]:
    """Write a video whose motion schedule is known, and return it in ms.

    The scene only moves inside `motion_windows`, so any frame-selection
    strategy can be scored against a labelled ground truth rather than against
    a second heuristic.

    With `camera_pan` (the default) the whole frame translates across a
    textured backdrop, the way a handheld capture does. A moving subject alone
    cannot do this: its motion score is bounded by its own area, so it never
    reaches the sampler's 0.18 high-motion threshold no matter how fast it
    travels, and the high-motion branch would go unexercised.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 320, 240
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),  # ty: ignore[unresolved-attribute]
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Unable to create synthetic motion video")

    texture = _panning_texture(width * 4, height * 2, seed) if camera_pan else None
    total_frames = int(fps * seconds)
    x = 40
    pan_x = 0
    pan_y = 0
    try:
        for index in range(total_frames):
            elapsed = index / fps
            moving = any(start <= elapsed < end for start, end in motion_windows)
            if moving:
                x = (x + speed_px_per_frame) % (width - 60)
                pan_x += speed_px_per_frame
                pan_y += speed_px_per_frame // 3

            if texture is not None:
                origin_x = pan_x % (texture.shape[1] - width)
                origin_y = pan_y % (texture.shape[0] - height)
                frame = texture[
                    origin_y : origin_y + height, origin_x : origin_x + width
                ].copy()
            else:
                frame = np.full((height, width, 3), (18, 28, 36), np.uint8)

            cv2.rectangle(frame, (x, 90), (x + 50, 140), (70, 210, 160), -1)
            writer.write(frame)
    finally:
        writer.release()

    return [(start * 1000.0, end * 1000.0) for start, end in motion_windows]
