from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from spatial_ingestion.metadata.schema import SourceType, Track
from spatial_ingestion.resource_tagging.priority import LatencyAwareResourceTagger

IMAGE_MIME_PREFIX = "image/"
VIDEO_MIME_PREFIX = "video/"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@dataclass(frozen=True)
class MediaItemDescriptor:
    filename: str
    mime_type: str | None = None
    size_bytes: int | None = None


@dataclass(frozen=True)
class RoutingDecision:
    input_type: SourceType
    track: Track
    priority_score: float
    reason: str


class MediaClassifierRouter:
    """Dynamic decision matrix for static files, folders, and live streams."""

    def __init__(self, tagger: LatencyAwareResourceTagger | None = None) -> None:
        self._tagger = tagger or LatencyAwareResourceTagger()

    def classify_static(self, items: list[MediaItemDescriptor]) -> RoutingDecision:
        if not items:
            return self._decision(SourceType.UNKNOWN, Track.BATCH, 0, "empty payload")

        kinds_by_file = {item.filename: self._kind_for_item(item) for item in items}
        media_kinds = set(kinds_by_file.values())

        if SourceType.UNKNOWN in media_kinds:
            unknown_files = [
                filename
                for filename, kind in kinds_by_file.items()
                if kind == SourceType.UNKNOWN
            ]
            return self._decision(
                SourceType.UNKNOWN,
                Track.BATCH,
                len(items),
                f"unsupported file(s): {', '.join(unknown_files)}",
            )

        if media_kinds == {SourceType.SINGLE_IMAGE}:
            source = SourceType.SINGLE_IMAGE if len(items) == 1 else SourceType.IMAGE_FOLDER
        elif media_kinds == {SourceType.SINGLE_VIDEO}:
            source = SourceType.SINGLE_VIDEO if len(items) == 1 else SourceType.VIDEO_FOLDER
        else:
            source = SourceType.UNKNOWN

        return self._decision(
            source,
            Track.BATCH,
            len(items),
            f"{len(items)} static item(s), kinds={sorted(kind.value for kind in media_kinds)}",
        )

    def classify_stream(self, transport: str, stream_id: str | None = None) -> RoutingDecision:
        normalized = transport.lower()
        if normalized != "websocket":
            return self._decision(
                SourceType.UNKNOWN,
                Track.LIVE,
                1,
                f"unsupported live transport={transport}; implemented transport=websocket",
            )

        return self._decision(
            SourceType.LIVE_STREAM,
            Track.LIVE,
            1,
            f"{normalized} live stream {stream_id or ''}".strip(),
        )

    def _decision(
        self,
        source_type: SourceType,
        track: Track,
        file_count: int,
        reason: str,
    ) -> RoutingDecision:
        return RoutingDecision(
            input_type=source_type,
            track=track,
            priority_score=self._tagger.score(source_type, track, file_count=file_count),
            reason=reason,
        )

    @staticmethod
    def _kind_for_item(item: MediaItemDescriptor) -> SourceType:
        mime = (item.mime_type or "").lower()
        ext = Path(item.filename).suffix.lower()

        if mime.startswith(IMAGE_MIME_PREFIX) or ext in IMAGE_EXTENSIONS:
            return SourceType.SINGLE_IMAGE
        if mime.startswith(VIDEO_MIME_PREFIX) or ext in VIDEO_EXTENSIONS:
            return SourceType.SINGLE_VIDEO
        return SourceType.UNKNOWN
