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

from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree

__all__ = ["AlignmentResult", "apply_alignment", "umeyama", "align_to_reference"]


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


def align_to_reference(
    source: np.ndarray,
    target: np.ndarray,
    *,
    with_scale: bool = False,
    iterations: int = 30,
    tolerance: float = 1e-7,
) -> tuple[np.ndarray, AlignmentResult]:
    """Nearest-neighbour ICP onto `target`, seeded by centroid alignment.

    Correspondences are unknown between a reconstruction and a GT point cloud,
    so `umeyama` cannot be used directly. This runs plain point-to-point ICP;
    it is deliberately simple and its residual is reported rather than trusted.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    tree = KDTree(target)

    current = source - source.mean(axis=0) + target.mean(axis=0)
    alignment = umeyama(source[:3], source[:3], with_scale=with_scale)
    previous_rmse = float("inf")

    for _ in range(iterations):
        _, indices = tree.query(current, k=1)
        alignment = umeyama(source, target[np.asarray(indices, dtype=int)], with_scale=with_scale)
        current = apply_alignment(source, alignment)
        if abs(previous_rmse - alignment.rmse) < tolerance:
            break
        previous_rmse = alignment.rmse

    return current, alignment
