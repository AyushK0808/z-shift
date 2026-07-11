from __future__ import annotations

from collections import defaultdict

from spatial_ingestion.metadata.schema import FrameReference, SyncMapEntry


class MultiSourceSyncer:
    """Aligns video-folder frames by nearest timestamps across source ids."""

    def build_sync_map(
        self,
        frames_by_source: dict[str, list[FrameReference]],
        sync_group_id: str,
        tolerance_ms: float = 120.0,
    ) -> list[SyncMapEntry]:
        usable = {
            source: sorted(
                [frame for frame in frames if frame.timestamp_ms is not None],
                key=lambda frame: frame.timestamp_ms or 0.0,
            )
            for source, frames in frames_by_source.items()
        }
        usable = {source: frames for source, frames in usable.items() if frames}
        if len(usable) < 2:
            return []

        anchor_source = max(usable, key=lambda source: len(usable[source]))
        sync_entries: list[SyncMapEntry] = []

        for anchor_frame in usable[anchor_source]:
            anchor_ts = anchor_frame.timestamp_ms or 0.0
            aligned_frames = {anchor_source: anchor_frame.index}
            offsets_ms: dict[str, float] = {}

            for source, frames in usable.items():
                if source == anchor_source:
                    continue
                nearest = min(
                    frames,
                    key=lambda frame: abs((frame.timestamp_ms or 0.0) - anchor_ts),
                )
                offset = (nearest.timestamp_ms or 0.0) - anchor_ts
                if abs(offset) <= tolerance_ms:
                    aligned_frames[source] = nearest.index
                    offsets_ms[source] = round(offset, 3)

            if len(aligned_frames) == len(usable):
                sync_entries.append(
                    SyncMapEntry(
                        sync_group_id=sync_group_id,
                        anchor_timestamp_ms=round(anchor_ts, 3),
                        aligned_frames=aligned_frames,
                        offsets_ms=offsets_ms,
                    )
                )

        return self._dedupe_entries(sync_entries)

    @staticmethod
    def _dedupe_entries(entries: list[SyncMapEntry]) -> list[SyncMapEntry]:
        seen: defaultdict[str, set[int]] = defaultdict(set)
        deduped: list[SyncMapEntry] = []
        for entry in entries:
            conflict = any(
                frame_index in seen[source]
                for source, frame_index in entry.aligned_frames.items()
            )
            if conflict:
                continue
            for source, frame_index in entry.aligned_frames.items():
                seen[source].add(frame_index)
            deduped.append(entry)
        return deduped

