"""P3 - fixture generators: controlled mesh damage and labelled media.

Synthetic *media* generation stays in
`spatial_ingestion.test_harness.media_factory` so there is one place that
knows how to write a synthetic video; this module wraps it and adds the
mesh-corruption and clock-offset fixtures the benchmarks need.
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import trimesh

from spatial_ingestion.metadata.schema import FrameReference
from spatial_ingestion.test_harness.media_factory import create_motion_video

__all__ = [
    "FACES_PER_HOLE",
    "corrupt_mesh",
    "derive_offset_stream",
    "frames_from_video",
    "make_motion_video",
]

FACES_PER_HOLE = 20


def _remove_face_patches(
    mesh: trimesh.Trimesh, n_patches: int, patch_size: int, rng: np.random.Generator
) -> trimesh.Trimesh:
    """Delete `n_patches` *connected* face patches, producing real holes.

    Dropping scattered random faces would produce `n_patches * patch_size`
    single-triangle punctures instead of `n_patches` holes, which is a
    different (and much easier) repair problem than the one hole filling is
    claimed to solve.
    """
    adjacency: dict[int, list[int]] = defaultdict(list)
    for left, right in np.asarray(mesh.face_adjacency, dtype=int):
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))

    n_faces = len(mesh.faces)
    seeds = rng.choice(n_faces, size=min(n_patches, n_faces), replace=False)
    dropped: set[int] = set()
    for seed_face in seeds:
        patch: set[int] = set()
        frontier: deque[int] = deque([int(seed_face)])
        while frontier and len(patch) < patch_size:
            face = frontier.popleft()
            if face in patch:
                continue
            patch.add(face)
            frontier.extend(n for n in adjacency[face] if n not in patch)
        dropped |= patch

    keep = np.setdiff1d(np.arange(n_faces), np.fromiter(dropped, dtype=int, count=len(dropped)))
    if keep.size == 0:
        raise ValueError("corruption removed every face; lower n_holes")
    damaged = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces)[keep], process=False
    )
    damaged.remove_unreferenced_vertices()
    return damaged


def corrupt_mesh(
    mesh: trimesh.Trimesh,
    *,
    noise_sigma: float = 0.0,
    n_holes: int = 0,
    n_fragments: int = 0,
    seed: int = 0,
) -> trimesh.Trimesh:
    """Apply controlled, *known* damage so refinement can be scored against the original.

    `noise_sigma` is a fraction of the bounding diagonal, so the same value
    means the same relative damage on any input.
    """
    rng = np.random.default_rng(seed)
    damaged = mesh.copy()
    bounds = np.asarray(damaged.bounds, dtype=float)
    diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))

    if noise_sigma:
        damaged.vertices = np.asarray(damaged.vertices) + rng.normal(
            0.0, noise_sigma * diagonal, np.asarray(damaged.vertices).shape
        )

    if n_holes:
        damaged = _remove_face_patches(damaged, n_holes, FACES_PER_HOLE, rng)

    if n_fragments:
        parts = [damaged]
        for _ in range(n_fragments):
            fragment = trimesh.creation.icosphere(subdivisions=1, radius=diagonal * 0.02)
            fragment.apply_translation(rng.uniform(-diagonal / 2, diagonal / 2, 3))
            parts.append(fragment)
        damaged = trimesh.util.concatenate(parts)

    return damaged


def make_motion_video(
    path: Path | str,
    *,
    fps: int = 24,
    seconds: int = 20,
    motion_windows: tuple[tuple[float, float], ...] = ((5.0, 9.0), (13.0, 16.0)),
    speed_px_per_frame: int = 6,
    camera_pan: bool = True,
    seed: int = 0,
) -> list[tuple[float, float]]:
    """Video with a *labelled* motion schedule. Returns the ground-truth windows in ms."""
    return create_motion_video(
        Path(path),
        fps=fps,
        seconds=seconds,
        motion_windows=motion_windows,
        speed_px_per_frame=speed_px_per_frame,
        camera_pan=camera_pan,
        seed=seed,
    )


def frames_from_video(
    video_path: Path | str,
    *,
    source_id: str = "cam_a",
    sampler: object | None = None,
) -> list[FrameReference]:
    """Sample a video through the production sampler into `FrameReference`s.

    Goes through `MotionAdaptiveFrameSampler` rather than reimplementing
    selection, so the experiments measure the shipped behaviour.
    """
    from spatial_ingestion.batch_normalization.video_sampler import (
        MotionAdaptiveFrameSampler,
    )

    active = sampler if sampler is not None else MotionAdaptiveFrameSampler()
    samples = active.sample(Path(video_path))  # ty: ignore[unresolved-attribute]
    base_uri = Path(video_path).resolve().as_uri()
    return [
        FrameReference(
            frame_id=f"{source_id}_{sample.index:06d}",
            uri=f"{base_uri}#frame={sample.index}",
            index=sample.index,
            timestamp_ms=sample.timestamp_ms,
            source_id=source_id,
            motion_score=sample.motion_score,
            resolution=(320, 240),
        )
        for sample in samples
    ]


def derive_offset_stream(
    src_frames: list[FrameReference],
    offset_ms: float,
    jitter_ms: float = 0.0,
    seed: int = 0,
    *,
    source_id: str | None = None,
    motion_noise: float = 0.0,
) -> list[FrameReference]:
    """Clone a frame list with a known injected clock offset. Ground truth = offset_ms.

    `motion_noise` perturbs each cloned frame's motion score. Without it the
    derived stream carries a byte-identical motion signature to the anchor,
    which makes `MultiSourceSyncer`'s matching trivial and leaves
    `MOTION_MATCH_TOLERANCE` unmeasurable; a second camera in a real rig sees
    the same event from a different viewpoint and scores it differently.
    """
    rng = np.random.default_rng(seed)
    derived: list[FrameReference] = []
    for frame in src_frames:
        clone = frame.model_copy(deep=True)
        base = frame.timestamp_ms if frame.timestamp_ms is not None else 0.0
        jitter = float(rng.normal(0.0, jitter_ms)) if jitter_ms else 0.0
        clone.timestamp_ms = base + offset_ms + jitter
        if motion_noise and frame.motion_score is not None:
            clone.motion_score = float(
                np.clip(frame.motion_score + rng.normal(0.0, motion_noise), 0.0, 1.0)
            )
        if source_id is not None:
            clone.source_id = source_id
            clone.frame_id = f"{source_id}_{frame.index:06d}"
        derived.append(clone)
    return derived
