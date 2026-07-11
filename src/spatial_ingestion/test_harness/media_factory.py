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
        cv2.VideoWriter_fourcc(*"mp4v"),
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

