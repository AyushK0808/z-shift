from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from spatial_ingestion.config import DEFAULT_IMAGE_SIZE, NORMALIZED_OUTPUT_ROOT
from spatial_ingestion.metadata.schema import FrameReference
from spatial_ingestion.batch_normalization.video_sampler import MotionAdaptiveFrameSampler


class VideoProcessor:
    def __init__(
        self,
        sampler: MotionAdaptiveFrameSampler | None = None,
        output_root: Path = NORMALIZED_OUTPUT_ROOT,
        target_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ) -> None:
        self._sampler = sampler or MotionAdaptiveFrameSampler()
        self._output_root = output_root
        self._target_size = target_size
        self._output_root.mkdir(parents=True, exist_ok=True)

    def extract_frames(
        self,
        video_path: Path,
        namespace: str,
        source_id: str | None = None,
        original_uri: str | None = None,
    ) -> list[FrameReference]:
        output_dir = self._output_root / namespace
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_refs: list[FrameReference] = []

        for sample_number, sample in enumerate(self._sampler.sample(video_path)):
            resized = self._resize_longest_side(sample.image)
            frame_id = f"frame_{uuid4().hex}"
            output_path = output_dir / f"{frame_id}.png"
            cv2.imwrite(str(output_path), resized)
            frame_refs.append(
                FrameReference(
                    frame_id=frame_id,
                    uri=output_path.as_uri(),
                    original_uri=original_uri,
                    index=sample_number,
                    timestamp_ms=sample.timestamp_ms,
                    source_id=source_id or video_path.stem,
                    motion_score=sample.motion_score,
                    resolution=(int(resized.shape[1]), int(resized.shape[0])),
                )
            )

        return frame_refs

    def _resize_longest_side(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        max_width, max_height = self._target_size
        scale = min(max_width / width, max_height / height, 1.0)
        if scale == 1.0:
            return image
        target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        return cv2.resize(image, target, interpolation=cv2.INTER_AREA)
