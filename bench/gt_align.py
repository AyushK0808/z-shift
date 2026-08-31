"""Rigid alignment to ground truth, for every Tier B comparison.

Reviewers look for two things in a GT comparison and reject its absence: which
alignment was used, and what the residual was. Both are returned here and both
land in the Tier B CSVs.

The default MASt3R checkpoint is the *metric* variant
(MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric), so scale should already be
roughly correct. `with_scale=False` is therefore the honest default: it tests
that claim instead of hiding a scale error inside the fit. Run both and report
which one the numbers came from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree

__all__ = ["AlignmentResult", "apply_alignment", "umeyama", "align_to_reference"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlignmentResult:
    rotation: np.ndarray
    translation: np.ndarray
    scale: float
    rmse: float
    n_correspondences: int
    with_scale: bool

    def as_row(self, prefix: str = "align_") -> dict[str, float | int | bool]:
        return {
            f"{prefix}scale": round(self.scale, 6),
            f"{prefix}rmse": round(self.rmse, 6),
            f"{prefix}n_correspondences": self.n_correspondences,
            f"{prefix}with_scale": self.with_scale,
            f"{prefix}translation_norm": round(float(np.linalg.norm(self.translation)), 6),
        }


def umeyama(source: np.ndarray, target: np.ndarray, with_scale: bool = False) -> AlignmentResult:
    """Least-squares similarity transform mapping `source` onto `target`.

    Umeyama (1991). Rows of `source` and `target` must correspond.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape:
        raise ValueError("source and target must have matching shapes")
    n = source.shape[0]
    if n < 3:
        raise ValueError("need at least 3 correspondences")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centred = source - source_mean
    target_centred = target - target_mean

    covariance = target_centred.T @ source_centred / n
    u, singular_values, vt = np.linalg.svd(covariance)

    # Guard against a reflection being fitted instead of a rotation.
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[2, 2] = -1.0
        singular_values = singular_values.copy()
        singular_values[2] *= -1.0

    rotation = u @ correction @ vt
    variance = float((source_centred**2).sum() / n)
    scale = float(singular_values.sum() / variance) if with_scale and variance > 0 else 1.0
    translation = target_mean - scale * rotation @ source_mean

    residuals = target - (scale * source @ rotation.T + translation)
    rmse = float(np.sqrt((residuals**2).sum(axis=1).mean()))
    return AlignmentResult(rotation, translation, scale, rmse, n, with_scale)


def apply_alignment(points: np.ndarray, alignment: AlignmentResult) -> np.ndarray:
    return alignment.scale * np.asarray(points, dtype=float) @ alignment.rotation.T + (
        alignment.translation
    )


def _principal_axes(points: np.ndarray) -> np.ndarray:
    """Right-handed principal-axis frame of `points`: columns, descending variance."""
    centred = points - points.mean(axis=0)
    _, vectors = np.linalg.eigh(centred.T @ centred)
    axes = vectors[:, ::-1]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return axes


# PCA fixes each axis only up to sign, so aligning source axes onto target axes
# is ambiguous; of the eight sign combinations, these four are the ones with
# determinant +1 (the other four are reflections, which umeyama already
# excludes from its own fit).
_AXIS_SIGN_FLIPS = (
    (1.0, 1.0, 1.0),
    (1.0, -1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
)


def _seed_rotations(source: np.ndarray, target: np.ndarray) -> list[np.ndarray]:
    """Candidate initial rotations to seed ICP from: identity, plus PCA-axis fits.

    Identity (translation-only seeding) is correct only when source and target
    already share an orientation. A self-calibrated reconstruction (MASt3R
    here) carries no relation to an external GT's world frame, so the true
    rotation can be arbitrarily large -- identity seeding then lands ICP in a
    bad local minimum on the very first nearest-neighbour query. PCA-axis
    alignment gives it a seed that is close to correct regardless of the true
    rotation, as long as the two clouds have comparably-shaped extents.
    """
    source_axes = _principal_axes(source)
    target_axes = _principal_axes(target)
    rotations = [np.eye(3)]
    for signs in _AXIS_SIGN_FLIPS:
        rotations.append(target_axes @ (source_axes * np.array(signs)).T)
    return rotations


def align_to_reference(
    source: np.ndarray,
    target: np.ndarray,
    *,
    with_scale: bool = False,
    iterations: int = 30,
    tolerance: float = 1e-7,
) -> tuple[np.ndarray, AlignmentResult]:
    """Nearest-neighbour ICP onto `target`, multi-seeded to escape rotation traps.

    Correspondences are unknown between a reconstruction and a GT point cloud,
    so `umeyama` cannot be used directly. Point-to-point ICP only finds the
    correct registration if it starts close to it -- a bad rotation seed
    converges to a confidently-wrong local minimum instead of failing loudly.
    So this runs the ICP loop once per candidate seed from `_seed_rotations`
    and keeps whichever run ends with the lowest RMSE; it is still plain
    point-to-point ICP underneath, and the winning residual is reported rather
    than trusted.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    tree = KDTree(target)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)

    seed_rotations = _seed_rotations(source, target)
    logger.info(
        "ICP: aligning %d points to %d GT points across %d seed rotations (<=%d iterations each)",
        len(source),
        len(target),
        len(seed_rotations),
        iterations,
    )

    best_current: np.ndarray | None = None
    best_alignment: AlignmentResult | None = None
    for seed_index, rotation in enumerate(seed_rotations):
        current = (source - source_mean) @ rotation.T + target_mean
        alignment = umeyama(source[:3], source[:3], with_scale=with_scale)
        previous_rmse = float("inf")

        for step in range(iterations):  # noqa: B007 (used after loop for logging)
            _, indices = tree.query(current, k=1)
            correspondences = target[np.asarray(indices, dtype=int)]
            alignment = umeyama(source, correspondences, with_scale=with_scale)
            current = apply_alignment(source, alignment)
            if abs(previous_rmse - alignment.rmse) < tolerance:
                break
            previous_rmse = alignment.rmse

        logger.info(
            "ICP seed %d/%d: converged after %d iterations, rmse=%.6f",
            seed_index + 1,
            len(seed_rotations),
            step + 1,
            alignment.rmse,
        )

        if best_alignment is None or alignment.rmse < best_alignment.rmse:
            best_current, best_alignment = current, alignment

    assert best_current is not None and best_alignment is not None
    return best_current, best_alignment
