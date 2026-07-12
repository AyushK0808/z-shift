from __future__ import annotations

from spatial_ingestion.config import DEFAULT_MULTI_VIEW_BACKEND
from spatial_ingestion.generation_handoff import GenerationHandoff, GenerationMode, HandoffFrame
from spatial_ingestion.reconstruction.models import ReconstructionJob, ReconstructionMode


class ReconstructionJobBuilder:
    def __init__(self, multi_view_backend: str = DEFAULT_MULTI_VIEW_BACKEND) -> None:
        self._multi_view_backend = multi_view_backend

    def build(self, handoff: GenerationHandoff) -> ReconstructionJob:
        if not handoff.supports_reconstruction:
            raise ValueError("handoff is not reconstruction-ready")

        if handoff.mode == GenerationMode.SINGLE_VIEW:
            raise ValueError("single-view backend selection is not implemented yet")
        if handoff.mode == GenerationMode.VIDEO_SEQUENCE:
            raise ValueError("video-sequence reconstruction is not implemented yet")
        if handoff.mode == GenerationMode.LIVE_STREAM:
            raise ValueError("live streams cannot be converted into reconstruction jobs")

        if handoff.mode == GenerationMode.MULTI_VIEW:
            frames = list(handoff.primary_frames)
            return ReconstructionJob(
                mode=ReconstructionMode.MULTI_VIEW,
                backend_name=self._multi_view_backend,
                image_uris=[frame.uri for frame in frames],
                frames=frames,
                metadata={
                    "source_type": handoff.source_type.value,
                    "track": handoff.track.value,
                },
                warnings=list(handoff.warnings),
            )

        if handoff.mode == GenerationMode.SYNCHRONIZED_VIEWS:
            frames = self._flatten_sync_groups(handoff.sync_view_groups)
            return ReconstructionJob(
                mode=ReconstructionMode.SYNCHRONIZED_VIEWS,
                backend_name=self._multi_view_backend,
                image_uris=[frame.uri for frame in frames],
                frames=frames,
                sync_view_groups=list(handoff.sync_view_groups),
                metadata={
                    "source_type": handoff.source_type.value,
                    "track": handoff.track.value,
                    "sync_group_id": handoff.sync_group_id,
                },
                warnings=list(handoff.warnings),
            )

        raise ValueError(f"Unsupported generation handoff mode: {handoff.mode}")

    @staticmethod
    def _flatten_sync_groups(sync_view_groups: list) -> list[HandoffFrame]:
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
