"""A5 - frame-budget selection and pair adjacency.

Supports (and tests) the SIV-C claim that "frames are ordered
deterministically by source, index, and identifier, and the job is capped at a
configurable frame budget; when the budget is exceeded, the highest-motion
frames are retained".

jobs.ReconstructionJobBuilder._cap_frames sorts by (motion_score, index)
descending and returns the top MAX_RECONSTRUCTION_FRAMES *without re-sorting
by index*, so above the budget job.image_uris are in motion-rank order, not
capture order. dust3r.image_pairs.make_pairs(scene_graph="swin") pairs purely
by list adjacency and is the default for video sequences -- exactly the case
where the budget overflows. This experiment measures what that costs in pair
adjacency, without needing MASt3R weights or a GPU.

Variants
    V1_as_implemented   current _cap_frames
    V2_motion_resort    same selection, re-sorted into capture order
    V3_uniform          every k-th frame, k = ceil(N / budget)
"""

from __future__ import annotations

import logging
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kendalltau

from bench.csvio import ResultWriter
from bench.fixtures import frames_from_video, make_motion_video
from bench.harness import experiment_parser, finish
from spatial_ingestion.config import MAX_RECONSTRUCTION_FRAMES
from spatial_ingestion.metadata.schema import (
    FrameReference,
    SourceType,
    Track,
    UnifiedSpatialIngestionSchema,
)
from spatial_ingestion.reconstruction.jobs import ReconstructionJobBuilder
from spatial_ingestion.reconstruction.models import HandoffFrame

EXP_ID = "a5_frame_budget"
VARIANTS: tuple[str, ...] = ("V1_as_implemented", "V2_motion_resort", "V3_uniform")
STRATEGIES: tuple[str, ...] = ("swin", "complete")

# Motion schedules chosen so the sampler overflows the 40-frame budget by
# different margins; a single schedule would measure one operating point.
SCHEDULES: dict[str, tuple[tuple[float, float], ...]] = {
    "sparse_motion": ((6.0, 12.0), (20.0, 26.0)),
    "half_motion": ((2.0, 8.0), (11.0, 17.0), (21.0, 28.0)),
    "mostly_motion": ((1.0, 29.0),),
}

# 30 s at 24 fps with a 12 px/frame pan puts every schedule above the 40-frame
# budget (50-80 sampled frames), which is the only regime where _cap_frames
# runs at all. A shorter or slower clip makes all three variants identical and
# the experiment vacuous.
VIDEO_SECONDS = 30
VIDEO_FPS = 24
PAN_SPEED_PX = 12

logger = logging.getLogger(__name__)


def _to_handoff(frames: list[FrameReference]) -> list[HandoffFrame]:
    return [
        HandoffFrame(
            frame_id=frame.frame_id,
            uri=frame.uri or f"file:///synthetic/{frame.frame_id}",
            index=frame.index,
            source_id=frame.source_id,
            timestamp_ms=frame.timestamp_ms,
            motion_score=frame.motion_score,
            resolution=frame.resolution,
        )
        for frame in frames
    ]


def _payload(frames: list[FrameReference]) -> UnifiedSpatialIngestionSchema:
    return UnifiedSpatialIngestionSchema(
        source_type=SourceType.SINGLE_VIDEO,
        track=Track.BATCH,
        resolution=(320, 240),
        frame_count=len(frames),
        is_stream=False,
        compute_priority_score=0.5,
        frames=frames,
    )


def select_variant(
    variant: str, frames: list[FrameReference], budget: int = MAX_RECONSTRUCTION_FRAMES
) -> list[HandoffFrame]:
    """Produce the frame list each variant would hand to pairing."""
    if variant == "V1_as_implemented":
        # Straight through the production builder, so this is the shipped path.
        return list(ReconstructionJobBuilder().build(_payload(frames)).frames)

    if variant == "V2_motion_resort":
        capped = ReconstructionJobBuilder().build(_payload(frames)).frames
        return sorted(capped, key=lambda f: (f.source_id or "", f.index))

    if variant == "V3_uniform":
        ordered = sorted(frames, key=lambda f: (f.source_id or "", f.index, f.frame_id))
        step = max(1, math.ceil(len(ordered) / budget))
        return _to_handoff(ordered[::step][:budget])

    raise ValueError(f"unknown variant '{variant}'")


def _pair_stats(selected: list[HandoffFrame], strategy: str) -> dict[str, Any]:
    """Temporal-gap statistics over the pairs make_pairs would build."""
    from dust3r.image_pairs import make_pairs

    images = [
        {"idx": position, "capture_index": frame.index, "timestamp_ms": frame.timestamp_ms}
        for position, frame in enumerate(selected)
    ]
    pairs = make_pairs(images, scene_graph=strategy, symmetrize=True)

    # Symmetrisation duplicates every edge; dedupe so the distribution is over
    # distinct pairs rather than double-counting each one.
    unique: dict[tuple[int, int], tuple[dict[str, Any], dict[str, Any]]] = {}
    for left, right in pairs:
        key = (min(left["idx"], right["idx"]), max(left["idx"], right["idx"]))
        unique.setdefault(key, (left, right))

    index_gaps = np.array(
        [abs(a["capture_index"] - b["capture_index"]) for a, b in unique.values()], dtype=float
    )
    time_gaps = np.array(
        [abs((a["timestamp_ms"] or 0.0) - (b["timestamp_ms"] or 0.0)) for a, b in unique.values()],
        dtype=float,
    )
    if index_gaps.size == 0:
        return {"n_pairs": 0, "n_pairs_symmetrized": len(pairs)}

    return {
        "n_pairs": int(index_gaps.size),
        "n_pairs_symmetrized": len(pairs),
        "pair_index_gap_median": float(np.median(index_gaps)),
        "pair_index_gap_p95": float(np.percentile(index_gaps, 95)),
        "pair_index_gap_mean": float(index_gaps.mean()),
        "pair_index_gap_max": float(index_gaps.max()),
        "pair_time_gap_median_ms": float(np.median(time_gaps)),
        "pair_time_gap_p95_ms": float(np.percentile(time_gaps, 95)),
        "pair_time_gap_mean_ms": float(time_gaps.mean()),
        "frac_pairs_over_1s": float((time_gaps > 1000.0).mean()),
    }


def _selection_stats(
    selected: list[HandoffFrame], all_frames: list[FrameReference]
) -> dict[str, Any]:
    positions = np.arange(len(selected), dtype=float)
    capture_index = np.array([frame.index for frame in selected], dtype=float)
    timestamps = np.array([frame.timestamp_ms or 0.0 for frame in selected], dtype=float)
    all_timestamps = np.array([f.timestamp_ms or 0.0 for f in all_frames], dtype=float)

    tau = float("nan")
    if len(selected) > 1 and float(capture_index.std()) > 0:
        tau = float(kendalltau(positions, capture_index).statistic)

    source_span = float(all_timestamps.max() - all_timestamps.min())
    retained_span = float(timestamps.max() - timestamps.min()) if len(timestamps) else 0.0
    ordered_ts = np.sort(timestamps)
    largest_gap = float(np.diff(ordered_ts).max()) if ordered_ts.size > 1 else 0.0

    motion = np.array([f.motion_score for f in selected if f.motion_score is not None], dtype=float)
    return {
        "n_selected": len(selected),
        "kendall_tau_position_vs_index": tau,
        "temporal_coverage": retained_span / source_span if source_span > 0 else float("nan"),
        "largest_unsampled_gap_ms": largest_gap,
        "mean_motion_score": float(motion.mean()) if motion.size else float("nan"),
        "in_capture_order": bool(np.all(np.diff(capture_index) > 0)),
    }


def run(
    results_dir: Path | None = None,
    *,
    quick: bool = False,
    seed: int = 0,
    fps: int = VIDEO_FPS,
    seconds: int = VIDEO_SECONDS,
) -> ResultWriter:
    writer = ResultWriter(EXP_ID, results_dir)
    schedules = dict(list(SCHEDULES.items())[:1]) if quick else SCHEDULES
    strategies = ("swin",) if quick else STRATEGIES

    with tempfile.TemporaryDirectory(prefix="a5_") as tmp:
        tmp_dir = Path(tmp)
        for schedule_name, windows in schedules.items():
            video = tmp_dir / f"{schedule_name}.mp4"
            make_motion_video(
                video,
                fps=fps,
                seconds=seconds,
                motion_windows=windows,
                speed_px_per_frame=PAN_SPEED_PX,
                seed=seed,
            )
            frames = frames_from_video(video, source_id="cam_a")
            logger.info("%s: %d sampled frames", schedule_name, len(frames))

            for variant in VARIANTS:
                selected = select_variant(variant, frames)
                selection = _selection_stats(selected, frames)
                for strategy in strategies:
                    writer.add(
                        schedule=schedule_name,
                        motion_fraction=round(
                            sum(end - start for start, end in windows) / seconds, 3
                        ),
                        n_frames_sampled=len(frames),
                        budget=MAX_RECONSTRUCTION_FRAMES,
                        over_budget=len(frames) > MAX_RECONSTRUCTION_FRAMES,
                        variant=variant,
                        pairing_strategy=strategy,
                        seed=seed,
                        **selection,
                        **_pair_stats(selected, strategy),
                    )
    return writer


def main(argv: list[str] | None = None) -> int:
    args = experiment_parser(EXP_ID, __doc__ or "").parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    writer = run(results_dir=args.results_dir, quick=args.quick, seed=args.seed)
    finish(writer)

    header = f"  {'variant':<20}{'strategy':<10}{'tau':>7}{'gapP95':>8}{'>1s':>7}{'cover':>7}"
    print(header)
    for row in writer.rows:
        nan = float("nan")
        print(
            f"  {row['variant']:<20}{row['pairing_strategy']:<10}"
            f"{row['kendall_tau_position_vs_index']:>7.2f}"
            f"{row.get('pair_index_gap_p95', nan):>8.1f}"
            f"{row.get('frac_pairs_over_1s', nan):>7.2f}"
            f"{row['temporal_coverage']:>7.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
