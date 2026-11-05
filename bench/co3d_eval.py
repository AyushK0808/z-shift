"""Scoring a reconstruction against CO3D's reference point cloud.

CO3D is not a ground-truth benchmark. Its `pointcloud.ply` is a COLMAP/MVS
reconstruction of the sequence, so it is a pseudo-reference: it carries
COLMAP's errors, and a method that resembles COLMAP scores better than one
that is merely right. The single-sequence chunk also ships no camera
annotations, so the camera-centre fit DTU scoring uses is not available.
A reconstruction is instead registered to the reference (a fixed scale, then ICP), which
minimises the very distances being scored and so flatters the result; the
registration residual is reported next to the scores so a poor fit is visible.

The reference holds the subject only (the sequence's foreground), while a
reconstruction also holds its support surface. Accuracy is therefore scored
only over points inside the reference's padded bounding box, the same device as
DTU's observation mask, and the registration should be given an object-only
reconstruction (`MeshCleaningConfig(isolate_object=True)`).

Distances are in the reference's own units, which CO3D leaves arbitrary, so
tolerances and errors are fractions of the reference's bounding-box diagonal.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import KDTree

from bench.gt_align import AlignmentResult, align_to_reference, apply_alignment

__all__ = ["CO3D_TEDDYBEAR_URL", "Co3dReference", "fetch_reference", "register", "score_co3d"]

logger = logging.getLogger(__name__)

CO3D_TEDDYBEAR_URL = "https://dl.fbaipublicfiles.com/co3dv2_231130/teddybear_001_singlesequence.zip"
# F-score tolerances as fractions of the reference's bounding-box diagonal.
TAU_FRACTIONS: tuple[float, ...] = (0.01, 0.02)
# Points further than this fraction of the diagonal outside the reference's
# bounding box are the support surface or background, not the subject.
BOUNDS_MARGIN_FRAC = 0.10
# Multipliers on the RMS-radius scale. An isolated reconstruction is usually
# missing parts of the subject, which biases the radius ratio; these bracket it.
SCALE_MULTIPLIERS: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.25)
_REGISTRATION_ITERATIONS = 30
_REGISTRATION_SOURCE_POINTS = 100_000
_REGISTRATION_TARGET_POINTS = 300_000


class Co3dReference:
    """One sequence's reference cloud, with its distance queries cached."""

    def __init__(self, points: np.ndarray) -> None:
        self.points = np.asarray(points, dtype=float)
        self.lower = self.points.min(axis=0)
        self.upper = self.points.max(axis=0)
        self.diagonal = float(np.linalg.norm(self.upper - self.lower))
        self.tree = KDTree(self.points)

    @classmethod
    def from_ply(cls, path: Path, max_points: int = 500_000, seed: int = 0) -> Co3dReference:
        from bench.tier_b_common import load_gt_points

        return cls(load_gt_points(Path(path), max_points=max_points, seed=seed))

    def in_bounds(self, points: np.ndarray) -> np.ndarray:
        margin = BOUNDS_MARGIN_FRAC * self.diagonal
        return np.all((points >= self.lower - margin) & (points <= self.upper + margin), axis=1)


def fetch_reference(dest: Path, sequence: str, url: str = CO3D_TEDDYBEAR_URL) -> Path:
    """`<dest>/<sequence>/pointcloud.ply`, read out of the chunk by byte range if absent."""
    target = Path(dest) / sequence / "pointcloud.ply"
    if target.exists():
        return target
    from bench.fetch_dtu import HttpRangeFile, _extract

    name = f"teddybear/{sequence}/pointcloud.ply"
    logger.info("fetching %s from %s", name, url)
    with zipfile.ZipFile(HttpRangeFile(url)) as archive:
        try:
            info = archive.getinfo(name)
        except KeyError as exc:
            raise FileNotFoundError(f"{name} is not in {url}") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        _extract(archive, info, target)
    return target


def _rms_radius(points: np.ndarray) -> float:
    return float(np.sqrt(((points - points.mean(axis=0)) ** 2).sum(axis=1).mean()))


def _symmetric_chamfer(a: np.ndarray, b: np.ndarray, b_tree: KDTree) -> float:
    return 0.5 * float(
        b_tree.query(a, k=1, workers=-1)[0].mean() + KDTree(a).query(b, k=1, workers=-1)[0].mean()
    )


def register(
    source: np.ndarray, reference: Co3dReference, seed: int = 0
) -> tuple[AlignmentResult, dict[str, Any]]:
    """Register an object-only reconstruction to the reference; scale, then rigid ICP.

    ICP that fits its own scale collapses a partial reconstruction onto part of
    the reference (accuracy near zero, completeness terrible), so the scale is
    fixed first, from the ratio of the two clouds' RMS radii, and tried at a few
    multipliers around that. Each is registered rigidly with ICP from several
    rotation seeds, and the one with the lowest *symmetric* chamfer wins:
    shrinking onto a fragment cannot lower that.
    """
    rng = np.random.default_rng(seed)
    source = np.asarray(source, dtype=float)
    if len(source) > _REGISTRATION_SOURCE_POINTS:
        source = source[rng.choice(len(source), _REGISTRATION_SOURCE_POINTS, replace=False)]
    target = reference.points
    if len(target) > _REGISTRATION_TARGET_POINTS:
        target = target[rng.choice(len(target), _REGISTRATION_TARGET_POINTS, replace=False)]
    target_tree = KDTree(target)
    centre = source.mean(axis=0)
    base_scale = _rms_radius(target) / _rms_radius(source)

    best: tuple[float, AlignmentResult, float] | None = None
    for multiplier in SCALE_MULTIPLIERS:
        scale = base_scale * multiplier
        scaled = (source - centre) * scale
        aligned, rigid = align_to_reference(
            scaled, target, with_scale=False, iterations=_REGISTRATION_ITERATIONS, tree=target_tree
        )
        chamfer = _symmetric_chamfer(aligned, target, target_tree)
        if best is None or chamfer < best[0]:
            composed = AlignmentResult(
                rotation=rigid.rotation,
                translation=rigid.translation - scale * (rigid.rotation @ centre),
                scale=scale,
                rmse=rigid.rmse,
                n_correspondences=rigid.n_correspondences,
                with_scale=True,
            )
            best = (chamfer, composed, multiplier)
    assert best is not None
    chamfer, alignment, multiplier = best
    return alignment, {
        "registration_scale": round(alignment.scale, 6),
        "registration_scale_multiplier": multiplier,
        "registration_rmse_frac": round(alignment.rmse / reference.diagonal, 5),
        "registration_chamfer_frac": round(chamfer / reference.diagonal, 5),
    }


def score_co3d(
    points: np.ndarray, reference: Co3dReference, alignment: AlignmentResult
) -> dict[str, Any]:
    """Accuracy, completeness and F-scores of `points` once registered to the reference."""
    placed = apply_alignment(points, alignment)
    placed = placed[reference.in_bounds(placed)]
    if len(placed) == 0:
        return {"n_points_in_bounds": 0}
    to_reference = np.asarray(reference.tree.query(placed, k=1, workers=-1)[0], dtype=float)
    to_reconstruction = np.asarray(
        KDTree(placed).query(reference.points, k=1, workers=-1)[0], dtype=float
    )
    row: dict[str, Any] = {
        "n_points_in_bounds": int(len(placed)),
        "accuracy_frac": round(float(to_reference.mean()) / reference.diagonal, 5),
        "completeness_frac": round(float(to_reconstruction.mean()) / reference.diagonal, 5),
    }
    row["chamfer_frac"] = round(0.5 * (row["accuracy_frac"] + row["completeness_frac"]), 5)
    for fraction in TAU_FRACTIONS:
        tau = fraction * reference.diagonal
        precision = float((to_reference < tau).mean())
        recall = float((to_reconstruction < tau).mean())
        f_score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        key = f"{round(fraction * 100)}pct"
        row[f"precision_{key}"] = round(precision, 4)
        row[f"recall_{key}"] = round(recall, 4)
        row[f"f_score_{key}"] = round(f_score, 4)
    return row
