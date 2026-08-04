from __future__ import annotations

from spatial_ingestion.config import MAX_RECONSTRUCTION_FRAMES, SWIN_PAIRING_THRESHOLD
from spatial_ingestion.metadata.schema import (
    FrameReference,
    SourceType,
    UnifiedSpatialIngestionSchema,
)
from spatial_ingestion.reconstruction.models import (
    Mast3rRunParams,
    ReconstructionJob,
    ReconstructionMode,
    SyncViewGroup,
)

# SourceType -> (ReconstructionMode, default pairing strategy). The pairing
# default is only applied when the caller does not override it.
_MODE_FOR_SOURCE: dict[SourceType, tuple[ReconstructionMode, str | None]] = {
    SourceType.SINGLE_IMAGE: (ReconstructionMode.SINGLE_VIEW, None),
    SourceType.IMAGE_FOLDER: (ReconstructionMode.MULTI_VIEW, None),
    SourceType.SINGLE_VIDEO: (ReconstructionMode.VIDEO_SEQUENCE, "swin"),
    SourceType.VIDEO_FOLDER: (ReconstructionMode.SYNCHRONIZED_VIEWS, None),
}


class ReconstructionJobBuilder:
    """Converts a Phase 1 schema into a Phase 2 reconstruction job."""

    def __init__(self) -> None:
        pass

    def build(self, payload: UnifiedSpatialIngestionSchema) -> ReconstructionJob:
        if payload.source_type == SourceType.LIVE_STREAM:
            raise ValueError(
                "live streams cannot be converted into reconstruction jobs: Phase 1 Track B "
                "(live ingestion) has no Phase 2 handoff yet"
            )

        try:
            rec_mode, pairing_default = _MODE_FOR_SOURCE[payload.source_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported source type: {payload.source_type}") from exc

        if rec_mode == ReconstructionMode.SINGLE_VIEW:
            raise ValueError(
                f"{rec_mode.value} reconstruction is not implemented yet: source_type="
                f"'{payload.source_type.value}' carries a single frame. Provide at least two "
                "views (image_folder) or a video capture."
            )

        provenance: dict[str, object] = {
            "source_type": payload.source_type.value,
            "track": payload.track.value,
            "resolution": payload.resolution,
            "frame_count": payload.frame_count,
            "compute_priority_score": payload.compute_priority_score,
        }

        if rec_mode == ReconstructionMode.SYNCHRONIZED_VIEWS:
            sync_groups = _build_sync_view_groups(payload)
            sync_groups = self._cap_sync_groups(sync_groups)
            sync_frames = _flatten_sync_groups(sync_groups)
            provenance["sync_group_id"] = payload.sync_group_id
            return ReconstructionJob(
                mode=rec_mode,
                image_uris=[_require_uri(f) for f in sync_frames],
                frames=sync_frames,
                sync_view_groups=sync_groups,
                params=_params_with_pairing(pairing_default),
                metadata=provenance,
            )

        frames = self._ordered_frames(payload.frames)
        handoff_frames = self._cap_frames(frames)
        if pairing_default is None and len(payload.frames) > SWIN_PAIRING_THRESHOLD:
            pairing_default = "swin"
        return ReconstructionJob(
            mode=rec_mode,
            image_uris=[_require_uri(f) for f in handoff_frames],
            frames=handoff_frames,
            params=_params_with_pairing(pairing_default),
            metadata=provenance,
        )

    @staticmethod
    def _ordered_frames(frames: list[FrameReference]) -> list[FrameReference]:
        return sorted(frames, key=lambda f: (f.source_id or "", f.index, f.frame_id))

    @staticmethod
    def _cap_frames(frames: list[FrameReference]) -> list[FrameReference]:
        if len(frames) <= MAX_RECONSTRUCTION_FRAMES:
            return frames
        sorted_frames = sorted(
            frames,
            key=lambda f: (
                f.motion_score if f.motion_score is not None else float("-inf"),
                f.index,
            ),
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


def _params_with_pairing(pairing_default: str | None) -> Mast3rRunParams:
    if pairing_default is None:
        return Mast3rRunParams()
    return Mast3rRunParams(pairing_strategy=pairing_default)


def _require_uri(frame: FrameReference) -> str:
    if frame.uri is None:
        raise ValueError(f"Frame {frame.frame_id} is missing a normalized asset URI")
    return frame.uri


def _flatten_sync_groups(sync_view_groups: list[SyncViewGroup]) -> list[FrameReference]:
    seen: set[tuple[str, str]] = set()
    frames: list[FrameReference] = []
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
        (frame.source_id, frame.index): frame
        for frame in payload.frames
        if frame.source_id is not None
    }

    sync_groups: list[SyncViewGroup] = []
    for entry in payload.sync_map:
        frames_by_source: dict[str, FrameReference] = {}
        for source_id, frame_number in entry.aligned_frames.items():
            resolved = frame_index.get((source_id, frame_number))
            if resolved is None:
                raise ValueError(
                    "sync_map references a frame that is not present in the payload: "
                    f"{source_id}[{frame_number}]"
                )
            if resolved.uri is None:
                raise ValueError(f"Frame {resolved.frame_id} is missing a normalized asset URI")
            frames_by_source[source_id] = resolved

        sync_groups.append(
            SyncViewGroup(
                anchor_timestamp_ms=entry.anchor_timestamp_ms,
                frames_by_source=frames_by_source,
                offsets_ms=dict(entry.offsets_ms),
            )
        )

    # NOTE: `anchor_timestamp_ms` / `offsets_ms` are carried through to Phase 2
    # for traceability, but current pairing (`reconstruction.pairing`) uses
    # group membership only — MASt3R alignment does not consume timestamps, so
    # the clock-offset values do not change reconstruction behaviour today.
    return sync_groups
