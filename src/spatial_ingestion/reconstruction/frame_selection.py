from __future__ import annotations

from pathlib import Path


def from_video(
    video_path: Path,
    max_frames: int = 40,
    strategy: str = "uniform",
    **kwargs: object,
) -> list[Path]:
    raise NotImplementedError("Video frame selection is not yet implemented")
