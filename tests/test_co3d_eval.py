"""CO3D scoring: the registration must recover a known placement and must not collapse."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from bench.co3d_eval import Co3dReference, register, score_co3d
from bench.gt_align import apply_alignment


def _surface(n: int, seed: int) -> np.ndarray:
    """An uneven ellipsoid surface, in arbitrary units, so ICP cannot spin freely on it."""
    points = np.random.default_rng(seed).normal(size=(n, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points * [1.0, 0.6, 0.35] * 3.7 + [0.4, -1.0, 2.0]


def _reconstruction(points: np.ndarray, scale: float) -> np.ndarray:
    """The same surface in another frame and scale, as a reconstruction would be."""
    turn = Rotation.from_rotvec([0.4, -0.9, 0.3]).as_matrix()
    return scale * (points @ turn.T) + [5.0, 2.0, -3.0]


def test_registration_recovers_a_scaled_rotated_copy() -> None:
    reference = Co3dReference(_surface(20_000, seed=0))
    reconstruction = _reconstruction(_surface(15_000, seed=1), scale=0.23)

    alignment, row = register(reconstruction, reference)
    scores = score_co3d(reconstruction, reference, alignment)

    assert row["registration_chamfer_frac"] < 0.01
    assert scores["f_score_2pct"] > 0.95
    placed = apply_alignment(reconstruction, alignment)
    assert np.abs(placed.mean(axis=0) - reference.points.mean(axis=0)).max() < 0.2


def test_registration_does_not_shrink_a_partial_reconstruction_onto_a_fragment() -> None:
    reference = Co3dReference(_surface(20_000, seed=0))
    full = _surface(15_000, seed=1)
    half = full[full[:, 0] > full[:, 0].mean()]
    reconstruction = _reconstruction(half, scale=0.23)

    alignment, _ = register(reconstruction, reference)
    scores = score_co3d(reconstruction, reference, alignment)

    # The true scale is 1/0.23. A collapse onto a fragment would have chosen a
    # far smaller one, so the fitted scale has to stay near it.
    assert alignment.scale * 0.23 > 0.6
    assert scores["accuracy_frac"] < 0.05


def test_points_outside_the_reference_bounds_are_not_scored() -> None:
    reference = Co3dReference(_surface(5_000, seed=0))
    floor = reference.points.copy()
    floor[:, 2] = reference.upper[2] + 10 * reference.diagonal
    alignment, _ = register(reference.points, reference)

    assert score_co3d(floor, reference, alignment)["n_points_in_bounds"] == 0
