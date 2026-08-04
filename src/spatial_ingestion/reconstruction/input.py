from __future__ import annotations

from pathlib import Path

from spatial_ingestion.media_classifier.router import IMAGE_EXTENSIONS


def collect_input_images(input_path: Path) -> list[Path]:
    """Collect supported image files from a folder, sorted by name."""
    if input_path.is_file():
        raise ValueError("Need a folder containing at least two views of the same subject")

    if input_path.is_dir():
        image_paths = sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise ValueError(f"No supported images found in directory: {input_path}")
        return image_paths

    raise FileNotFoundError(f"Input path does not exist: {input_path}")
