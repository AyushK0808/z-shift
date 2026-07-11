from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from PIL import Image

from spatial_ingestion.batch_normalization.exif import ExifExtractor
from spatial_ingestion.batch_normalization.ffmpeg_tools import FFmpegTools
from spatial_ingestion.batch_normalization.image_processor import ImageProcessor
from spatial_ingestion.batch_normalization.video_processor import VideoProcessor
from spatial_ingestion.media_classifier.router import RoutingDecision
from spatial_ingestion.metadata.schema import (
    CameraIntrinsics,
    FrameReference,
    SourceType,
    UnifiedSpatialIngestionSchema,
)
from spatial_ingestion.sync.multi_source import MultiSourceSyncer


class BatchNormalizer:
    def __init__(
        self,
        image_processor: ImageProcessor | None = None,
        video_processor: VideoProcessor | None = None,
        exif_extractor: ExifExtractor | None = None,
        ffmpeg: FFmpegTools | None = None,
        syncer: MultiSourceSyncer | None = None,
    ) -> None:
        self._images = image_processor or ImageProcessor()
        self._videos = video_processor or VideoProcessor()
        self._exif = exif_extractor or ExifExtractor()
        self._ffmpeg = ffmpeg or FFmpegTools()
        self._syncer = syncer or MultiSourceSyncer()

    def normalize(
        self,
        paths: list[Path],
        decision: RoutingDecision,
        sync_group_id: str | None = None,
    ) -> UnifiedSpatialIngestionSchema:
        namespace = f"ingest_{uuid4().hex}"
        source_type = decision.input_type

        if source_type in {SourceType.SINGLE_IMAGE, SourceType.IMAGE_FOLDER}:
            frames = self._normalize_images(paths, namespace)
            intrinsics = self._first_intrinsics(paths)
            resolution = frames[0].resolution if frames else None
            return UnifiedSpatialIngestionSchema(
                source_type=source_type,
                track=decision.track,
                resolution=resolution,
                frame_count=len(frames),
                is_stream=False,
                camera_intrinsics=intrinsics,
                compute_priority_score=decision.priority_score,
                sync_group_id=sync_group_id,
                frames=frames,
            )

        if source_type in {SourceType.SINGLE_VIDEO, SourceType.VIDEO_FOLDER}:
            frames_by_source: dict[str, list[FrameReference]] = {}
            ffmpeg_metadata = {}
            for path in paths:
                source_id = path.stem
                frames_by_source[source_id] = self._videos.extract_frames(
                    path,
                    namespace=namespace,
                    source_id=source_id,
                )
                ffmpeg_metadata[source_id] = self._ffmpeg.probe(path)

            frames = [frame for source_frames in frames_by_source.values() for frame in source_frames]
            group_id = sync_group_id
            sync_map = []
            if source_type == SourceType.VIDEO_FOLDER:
                group_id = group_id or UnifiedSpatialIngestionSchema.new_sync_group_id()
                sync_map = self._syncer.build_sync_map(frames_by_source, group_id)

            resolution = frames[0].resolution if frames else None
            return UnifiedSpatialIngestionSchema(
                source_type=source_type,
                track=decision.track,
                resolution=resolution,
                frame_count=len(frames),
                is_stream=False,
                camera_intrinsics=None,
                compute_priority_score=decision.priority_score,
                sync_group_id=group_id,
                frames=frames,
                sync_map=sync_map,
                metadata={
                    "ffmpeg_probe": ffmpeg_metadata,
                    "sampling": "motion_adaptive_frame_diff",
                },
            )

        raise ValueError(f"Unsupported batch source type: {source_type}")

    def _normalize_images(self, paths: list[Path], namespace: str) -> list[FrameReference]:
        return [
            self._images.normalize_image(path, namespace=namespace, index=index)
            for index, path in enumerate(paths)
        ]

    def _first_intrinsics(self, paths: list[Path]) -> CameraIntrinsics | None:
        for path in paths:
            try:
                with Image.open(path):
                    intrinsics = self._exif.extract(path)
                    if intrinsics.raw_exif or intrinsics.make or intrinsics.model:
                        return intrinsics
                    return intrinsics
            except Exception:
                continue
        return None

