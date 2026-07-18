from __future__ import annotations

from spatial_ingestion.config import DEFAULT_MULTI_VIEW_BACKEND, MAX_RECONSTRUCTION_FRAMES, SWIN_PAIRING_THRESHOLD
from spatial_ingestion.metadata.schema import FrameReference, SourceType, UnifiedSpatialIngestionSchema
from spatial_ingestion.reconstruction.models import (
    GenerationMode,
    HandoffFrame,
    ReconstructionJob,
    ReconstructionMode,
    SyncViewGroup,
)


class ReconstructionJobBuilder:
    def __init__(self, multi_view_backend: str = DEFAULT_MULTI_VIEW_BACKEND) -> None:
        self._multi_view_backend = multi_view_backend

    def build(self, payload: UnifiedSpatialIngestionSchema) -> ReconstructionJob:
        if payload.source_type == SourceType.LIVE_STREAM:
            raise ValueError("live streams cannot be converted into reconstruction jobs")

        mode = self._mode_for_source(payload.source_type)
        if mode == GenerationMode.SINGLE_VIEW:
            raise ValueError(f"{mode.value} reconstruction is not implemented yet")

        if mode == GenerationMode.MULTI_VIEW:
            frames = self._ordered_frames(payload.frames)
            handoff_frames = self._cap_frames([_to_handoff_frame(f) for f in frames])
            metadata: dict[str, object] = {
                "source_type": payload.source_type.value,
                "track": payload.track.value,
            }
            if len(payload.frames) > SWIN_PAIRING_THRESHOLD:
                metadata["pairing_strategy"] = "swin"
            return ReconstructionJob(
                mode=ReconstructionMode.MULTI_VIEW,
                backend_name=self._multi_view_backend,
                image_uris=[f.uri for f in handoff_frames],
                frames=handoff_frames,
                metadata=metadata,
            )

        if mode == GenerationMode.VIDEO_SEQUENCE:
            frames = self._ordered_frames(payload.frames)
            handoff_frames = self._cap_frames([_to_handoff_frame(f) for f in frames])
            return ReconstructionJob(
                mode=ReconstructionMode.VIDEO_SEQUENCE,
                backend_name=self._multi_view_backend,
                image_uris=[f.uri for f in handoff_frames],
                frames=handoff_frames,
                metadata={
                    "source_type": payload.source_type.value,
                    "track": payload.track.value,
                    "pairing_strategy": "swin",
                },
            )

        if mode == GenerationMode.SYNCHRONIZED_VIEWS:
            sync_groups = _build_sync_view_groups(payload)
            sync_frames = self._cap_frames(_flatten_sync_groups(sync_groups))
            return ReconstructionJob(
                mode=ReconstructionMode.SYNCHRONIZED_VIEWS,
                backend_name=self._multi_view_backend,
                image_uris=[f.uri for f in sync_frames],
                frames=sync_frames,
                sync_view_groups=sync_groups,
                metadata={
                    "source_type": payload.source_type.value,
                    "track": payload.track.value,
                    "sync_group_id": payload.sync_group_id,
                },
            )

        raise ValueError(f"Unsupported source type: {payload.source_type}")

    @staticmethod
    def _mode_for_source(source_type: SourceType) -> GenerationMode:
        mapping = {
            SourceType.SINGLE_IMAGE: GenerationMode.SINGLE_VIEW,
            SourceType.IMAGE_FOLDER: GenerationMode.MULTI_VIEW,
            SourceType.SINGLE_VIDEO: GenerationMode.VIDEO_SEQUENCE,
            SourceType.VIDEO_FOLDER: GenerationMode.SYNCHRONIZED_VIEWS,
        }
        try:
            return mapping[source_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported source type: {source_type}") from exc

    @staticmethod
    def _ordered_frames(frames: list[FrameReference]) -> list[FrameReference]:
        return sorted(frames, key=lambda f: (f.source_id or "", f.index, f.frame_id))

    @staticmethod
    def _cap_frames(frames: list[HandoffFrame]) -> list[HandoffFrame]:
        if len(frames) <= MAX_RECONSTRUCTION_FRAMES:
            return frames
        sorted_frames = sorted(
            frames,
            key=lambda f: (f.motion_score if f.motion_score is not None else float("-inf"), f.index),
            reverse=True,
        )
        return sorted_frames[:MAX_RECONSTRUCTION_FRAMES]


def _to_handoff_frame(frame: FrameReference) -> HandoffFrame:
    if not frame.uri:
        raise ValueError(f"Frame {frame.frame_id} is missing a normalized asset URI")

    return HandoffFrame(
        frame_id=frame.frame_id,
        uri=frame.uri,
        index=frame.index,
        source_id=frame.source_id,
        timestamp_ms=frame.timestamp_ms,
        motion_score=frame.motion_score,
        resolution=frame.resolution,
        camera_intrinsics=frame.camera_intrinsics,
    )


def _flatten_sync_groups(sync_view_groups: list[SyncViewGroup]) -> list[HandoffFrame]:
    seen: set[str] = set()
    frames: list[HandoffFrame] = []
    for group in sync_view_groups:
        for source_id in sorted(group.frames_by_source):
            frame = group.frames_by_source[source_id]
            if frame.frame_id in seen:
                continue
            seen.add(frame.frame_id)
            frames.append(frame)
    return frames


def _build_sync_view_groups(payload: UnifiedSpatialIngestionSchema) -> list[SyncViewGroup]:
    frame_index = {
        (frame.source_id, frame.index): _to_handoff_frame(frame)
        for frame in payload.frames
        if frame.source_id is not None
    }

    sync_groups: list[SyncViewGroup] = []
    for entry in payload.sync_map:
        frames_by_source: dict[str, HandoffFrame] = {}
        for source_id, frame_number in entry.aligned_frames.items():
            resolved = frame_index.get((source_id, frame_number))
            if resolved is None:
                raise ValueError(
                    "sync_map references a frame that is not present in the payload: "
                    f"{source_id}[{frame_number}]"
                )
            frames_by_source[source_id] = resolved

        sync_groups.append(
            SyncViewGroup(
                anchor_timestamp_ms=entry.anchor_timestamp_ms,
                frames_by_source=frames_by_source,
                offsets_ms=dict(entry.offsets_ms),
            )
        )

    return sync_groups
