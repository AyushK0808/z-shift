from __future__ import annotations

from spatial_ingestion.metadata.schema import SourceType, Track


class LatencyAwareResourceTagger:
    """Assigns normalized compute priority scores at ingestion time."""

    def score(
        self,
        source_type: SourceType,
        track: Track,
        file_count: int = 1,
        estimated_frames: int | None = None,
    ) -> float:
        if track == Track.LIVE:
            return 1.0

        base_by_type = {
            SourceType.SINGLE_IMAGE: 0.55,
            SourceType.IMAGE_FOLDER: 0.35,
            SourceType.SINGLE_VIDEO: 0.45,
            SourceType.VIDEO_FOLDER: 0.25,
        }
        base = base_by_type.get(source_type, 0.2)

        if file_count > 1:
            base -= min(0.15, (file_count - 1) * 0.02)
        if estimated_frames and estimated_frames > 300:
            base -= min(0.10, (estimated_frames - 300) / 6000)

        return round(max(0.05, min(0.95, base)), 3)

