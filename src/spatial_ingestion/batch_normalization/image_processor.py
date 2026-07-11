from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps

from spatial_ingestion.config import DEFAULT_IMAGE_SIZE, NORMALIZED_OUTPUT_ROOT
from spatial_ingestion.metadata.schema import FrameReference


class ImageProcessor:
    def __init__(
        self,
        output_root: Path = NORMALIZED_OUTPUT_ROOT,
        target_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ) -> None:
        self._output_root = output_root
        self._target_size = target_size
        self._output_root.mkdir(parents=True, exist_ok=True)

    def normalize_image(
        self,
        image_path: Path,
        namespace: str,
        index: int = 0,
        source_id: str | None = None,
    ) -> FrameReference:
        output_dir = self._output_root / namespace
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_id = f"frame_{uuid4().hex}"
        output_path = output_dir / f"{frame_id}.jpg"

        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail(self._target_size, Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", self._target_size, (0, 0, 0))
            offset = (
                (self._target_size[0] - image.width) // 2,
                (self._target_size[1] - image.height) // 2,
            )
            canvas.paste(image, offset)
            canvas.save(output_path, format="JPEG", quality=92)

        return FrameReference(
            frame_id=frame_id,
            uri=output_path.as_uri(),
            index=index,
            source_id=source_id or image_path.stem,
            resolution=self._target_size,
        )

