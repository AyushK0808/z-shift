from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import cv2

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
    ) -> list[FrameReference]:
        output_dir = self._output_root / namespace
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_refs: list[FrameReference] = []

        for sample_number, sample in enumerate(self._sampler.sample(video_path)):
            resized = cv2.resize(sample.image, self._target_size, interpolation=cv2.INTER_AREA)
            frame_id = f"frame_{uuid4().hex}"
            output_path = output_dir / f"{frame_id}.jpg"
            cv2.imwrite(str(output_path), resized)
            frame_refs.append(
                FrameReference(
                    frame_id=frame_id,
                    uri=output_path.as_uri(),
                    index=sample_number,
                    timestamp_ms=sample.timestamp_ms,
                    source_id=source_id or video_path.stem,
                    motion_score=sample.motion_score,
                    resolution=self._target_size,
                )
            )

        return frame_refs

