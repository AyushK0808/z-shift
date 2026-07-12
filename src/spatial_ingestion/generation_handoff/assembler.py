from __future__ import annotations

from spatial_ingestion.generation_handoff.models import (
    GenerationHandoff,
    GenerationMode,
    HandoffFrame,
    SyncViewGroup,
)
from spatial_ingestion.metadata.schema import FrameReference, SourceType, UnifiedSpatialIngestionSchema


class GenerationHandoffBuilder:
    """Builds a generation-ready view of normalized ingestion payloads."""

    def build(self, payload: UnifiedSpatialIngestionSchema) -> GenerationHandoff:
        primary_frames = [self._to_handoff_frame(frame) for frame in self._ordered_frames(payload.frames)]
        mode = self._mode_for_source(payload.source_type)
        supports_reconstruction = payload.source_type != SourceType.LIVE_STREAM
        warnings: list[str] = []

        sync_view_groups: list[SyncViewGroup] = []
        if payload.source_type == SourceType.VIDEO_FOLDER:
            sync_view_groups = self._build_sync_view_groups(payload)
            if not sync_view_groups:
                warnings.append("video folder payload has no synchronized frame groups")

        if payload.source_type == SourceType.LIVE_STREAM:
            warnings.append("live stream payloads are not reconstruction-ready in Phase 2")

        return GenerationHandoff(
            source_type=payload.source_type,
            track=payload.track,
            mode=mode,
            supports_reconstruction=supports_reconstruction,
            primary_frames=primary_frames,
            sync_view_groups=sync_view_groups,
            sync_group_id=payload.sync_group_id,
            metadata=dict(payload.metadata),
            warnings=warnings,
        )

    @staticmethod
    def _mode_for_source(source_type: SourceType) -> GenerationMode:
        mapping = {
            SourceType.SINGLE_IMAGE: GenerationMode.SINGLE_VIEW,
            SourceType.IMAGE_FOLDER: GenerationMode.MULTI_VIEW,
            SourceType.SINGLE_VIDEO: GenerationMode.VIDEO_SEQUENCE,
            SourceType.VIDEO_FOLDER: GenerationMode.SYNCHRONIZED_VIEWS,
            SourceType.LIVE_STREAM: GenerationMode.LIVE_STREAM,
        }
        try:
            return mapping[source_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported source type for generation handoff: {source_type}") from exc

    @staticmethod
    def _ordered_frames(frames: list[FrameReference]) -> list[FrameReference]:
        return sorted(frames, key=lambda frame: (frame.source_id or "", frame.index, frame.frame_id))

    def _build_sync_view_groups(
        self,
        payload: UnifiedSpatialIngestionSchema,
    ) -> list[SyncViewGroup]:
        frame_index = {
            (frame.source_id, frame.index): self._to_handoff_frame(frame)
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

    @staticmethod
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
        )
