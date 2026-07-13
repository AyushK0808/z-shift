from __future__ import annotations

from collections import defaultdict

from spatial_ingestion.metadata.schema import FrameReference, SyncMapEntry

MOTION_MATCH_TOLERANCE = 0.03
OFFSET_BUCKET_MS = 100.0
MIN_MOTION_VARIANCE = 0.0005


class MultiSourceSyncer:
    """Aligns video-folder frames by estimated cross-source timestamp offsets."""

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
        source_offsets = self._estimate_offsets_ms(usable, anchor_source)
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
                    key=lambda frame: abs(
                        ((frame.timestamp_ms or 0.0) + source_offsets.get(source, 0.0))
                        - anchor_ts
                    ),
                )
                adjusted_offset = ((nearest.timestamp_ms or 0.0) + source_offsets.get(source, 0.0)) - anchor_ts
                if abs(adjusted_offset) <= tolerance_ms:
                    aligned_frames[source] = nearest.index
                    offsets_ms[source] = round(source_offsets.get(source, 0.0), 3)

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

    def _estimate_offsets_ms(
        self,
        frames_by_source: dict[str, list[FrameReference]],
        anchor_source: str,
    ) -> dict[str, float]:
        anchor = frames_by_source[anchor_source]
        offsets = {anchor_source: 0.0}
        for source, frames in frames_by_source.items():
            if source == anchor_source:
                continue
            offsets[source] = self._best_motion_signature_offset(anchor, frames)
        return offsets

    @staticmethod
    def _best_motion_signature_offset(
        anchor_frames: list[FrameReference],
        candidate_frames: list[FrameReference],
    ) -> float:
        anchor_signal = [
            (frame.timestamp_ms or 0.0, frame.motion_score or 0.0)
            for frame in anchor_frames
        ]
        candidate_signal = [
            (frame.timestamp_ms or 0.0, frame.motion_score or 0.0)
            for frame in candidate_frames
        ]
        if len(anchor_signal) < 2 or len(candidate_signal) < 2:
            return 0.0
        if (
            MultiSourceSyncer._signal_variance(anchor_signal) < MIN_MOTION_VARIANCE
            or MultiSourceSyncer._signal_variance(candidate_signal) < MIN_MOTION_VARIANCE
        ):
            return 0.0

        raw_offsets = [
            anchor_ts - candidate_ts
            for anchor_ts, anchor_motion in anchor_signal
            for candidate_ts, candidate_motion in candidate_signal
            if abs(anchor_motion - candidate_motion) <= MOTION_MATCH_TOLERANCE
        ]
        if not raw_offsets:
            return 0.0

        buckets: dict[int, list[float]] = {}
        for offset in raw_offsets:
            bucket = int(round(offset / OFFSET_BUCKET_MS))
            buckets.setdefault(bucket, []).append(offset)

        best_bucket = max(
            buckets,
            key=lambda bucket: (len(buckets[bucket]), -abs(sum(buckets[bucket]) / len(buckets[bucket]))),
        )
        values = sorted(buckets[best_bucket])
        midpoint = len(values) // 2
        if len(values) % 2:
            return round(values[midpoint], 3)
        return round((values[midpoint - 1] + values[midpoint]) / 2.0, 3)

    @staticmethod
    def _signal_variance(signal: list[tuple[float, float]]) -> float:
        values = [motion for _, motion in signal]
        mean = sum(values) / len(values)
        return sum((value - mean) ** 2 for value in values) / len(values)

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
