from __future__ import annotations

from spatial_ingestion.config import MAX_RECONSTRUCTION_FRAMES, SWIN_PAIRING_THRESHOLD
from spatial_ingestion.metadata.schema import FrameReference, SourceType, UnifiedSpatialIngestionSchema
from spatial_ingestion.reconstruction.models import (
    GenerationMode,
    HandoffFrame,
    ReconstructionJob,
    ReconstructionMode,
    SyncViewGroup,
)


class ReconstructionJobBuilder:
    def __init__(self) -> None:
        pass

    def build(self, payload: UnifiedSpatialIngestionSchema) -> ReconstructionJob:
        if payload.source_type == SourceType.LIVE_STREAM:
            raise ValueError("live streams cannot be converted into reconstruction jobs")

        mode = self._mode_for_source(payload.source_type)
        if mode == GenerationMode.SINGLE_VIEW:
            raise ValueError(f"{mode.value} reconstruction is not implemented yet")

        base_metadata: dict[str, object] = {
            "source_type": payload.source_type.value,
            "track": payload.track.value,
            "resolution": payload.resolution,
            "frame_count": payload.frame_count,
            "compute_priority_score": payload.compute_priority_score,
        }

        _MODE_MAP = {
            GenerationMode.MULTI_VIEW: (ReconstructionMode.MULTI_VIEW, None),
            GenerationMode.VIDEO_SEQUENCE: (ReconstructionMode.VIDEO_SEQUENCE, "swin"),
            GenerationMode.SYNCHRONIZED_VIEWS: (ReconstructionMode.SYNCHRONIZED_VIEWS, None),
        }
        rec_mode, pairing_default = _MODE_MAP[mode]

        if mode == GenerationMode.SYNCHRONIZED_VIEWS:
            sync_groups = _build_sync_view_groups(payload)
            sync_groups = self._cap_sync_groups(sync_groups)
            sync_frames = _flatten_sync_groups(sync_groups)
            metadata = dict(base_metadata)
            metadata["sync_group_id"] = payload.sync_group_id
            return ReconstructionJob(
                mode=rec_mode,
                image_uris=[f.uri for f in sync_frames],
                frames=sync_frames,
                sync_view_groups=sync_groups,
                metadata=metadata,
            )

        frames = self._ordered_frames(payload.frames)
        handoff_frames = self._cap_frames([_to_handoff_frame(f) for f in frames])
        metadata = dict(base_metadata)
        if pairing_default:
            metadata["pairing_strategy"] = pairing_default
        elif len(payload.frames) > SWIN_PAIRING_THRESHOLD:
            metadata["pairing_strategy"] = "swin"
        return ReconstructionJob(
            mode=rec_mode,
            image_uris=[f.uri for f in handoff_frames],
            frames=handoff_frames,
            metadata=metadata,
        )

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

    @staticmethod
    def _cap_sync_groups(groups: list[SyncViewGroup]) -> list[SyncViewGroup]:
        if not groups:
            return groups
        avg_cameras = sum(len(g.frames_by_source) for g in groups) / len(groups)
        max_groups = max(1, int(MAX_RECONSTRUCTION_FRAMES / avg_cameras))
        if len(groups) <= max_groups:
            return groups
        scored = sorted(
            groups,
            key=_group_motion_score,
            reverse=True,
        )
        return scored[:max_groups]


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
    seen: set[tuple[str, str]] = set()
    frames: list[HandoffFrame] = []
    for group in sync_view_groups:
        for source_id in sorted(group.frames_by_source):
            frame = group.frames_by_source[source_id]
            key = (source_id or "", frame.frame_id)
            if key in seen:
                continue
            seen.add(key)
            frames.append(frame)
    return frames


def _group_motion_score(group: SyncViewGroup) -> tuple[float, int]:
    best = float("-inf")
    best_index = 0
    for source_id in sorted(group.frames_by_source):
        frame = group.frames_by_source[source_id]
        score = frame.motion_score if frame.motion_score is not None else float("-inf")
        if score > best or (score == best and frame.index > best_index):
            best = score
            best_index = frame.index
    return (best, best_index)


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
