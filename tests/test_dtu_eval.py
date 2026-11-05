"""DTU's evaluation protocol on a synthetic scan with a known answer.

The reconstruction is the ground-truth surface plus a background slab, moved
into an arbitrary frame at 1/800 scale. That is the situation that broke ICP
from scratch: background the ground truth lacks, and a large scale error. The
camera fit has to undo both exactly, and the masked scores have to read as a
perfect reconstruction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.io
import trimesh
from scipy.spatial.transform import Rotation

from bench.dtu_eval import (
    DtuReference,
    camera_alignment,
    estimate_normals,
    load_reconstruction_cameras,
    mesh_normal_consistency,
    score_dtu,
    view_number,
)
from bench.tier_b_common import SceneSet, score_against_gt

SCALE = 800.0  # DTU mm per reconstruction unit
N_VIEWS = 12


def _surface(n: int, seed: int) -> np.ndarray:
    """An uneven ellipsoid: unlike a sphere, ICP cannot spin freely on it."""
    points = np.random.default_rng(seed).normal(size=(n, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points * [100.0, 70.0, 45.0] + [0, 0, 600.0]


def _write_scan(root: Path) -> tuple[DtuReference, np.ndarray, Path]:
    """A DTU-shaped scan on disk; returns its reference, a reconstruction, its output dir."""
    calibration = root / "Calibration" / "cal18"
    calibration.mkdir(parents=True)
    angles = np.linspace(0, 1.5 * np.pi, N_VIEWS)
    centres = np.stack([400 * np.cos(angles), 400 * np.sin(angles), 250 + 40 * angles], axis=1)
    intrinsics = np.array([[2900.0, 0, 800], [0, 2900.0, 600], [0, 0, 1]])
    for view, centre in enumerate(centres, start=1):
        rotation = Rotation.from_rotvec([0.1 * view, -0.2, 0.05]).as_matrix()
        projection = intrinsics @ np.hstack([rotation, (-rotation @ centre)[:, None]])
        np.savetxt(calibration / f"pos_{view:03d}.txt", projection)

    gt = _surface(60_000, seed=0)
    stl = root / "stl001_total.ply"
    trimesh.PointCloud(gt).export(stl)

    # Observation mask: the sphere's box, 2 mm voxels, every voxel observed.
    lo, hi = np.floor(gt.min(axis=0)) - 5, np.ceil(gt.max(axis=0)) + 5
    resolution = 2
    shape = np.ceil((hi - lo) / resolution).astype(int) + 1
    scipy.io.savemat(
        root / "ObsMask1_10.mat",
        {
            "ObsMask": np.ones(shape, dtype=np.uint8),
            "BB": np.stack([lo, hi]).astype(np.int16),
            "Res": np.array([[resolution]], dtype=np.uint8),
        },
    )
    # Table plane 1 mm below the sphere; ground truth above it counts.
    scipy.io.savemat(
        root / "Plane1.mat", {"P": np.array([[0.0], [0.0], [1.0], [-(gt[:, 2].min() - 1)]])}
    )
    reference = DtuReference(stl, calibration, root / "ObsMask1_10.mat", root / "Plane1.mat")

    # Reconstruction: a fresh sample of the surface plus a background slab far
    # outside the mask, in a frame rotated, shifted and shrunk by SCALE.
    to_recon = Rotation.from_rotvec([0.7, -1.9, 0.4]).as_matrix()
    offset = np.array([0.3, -1.2, 2.0])

    def into_reconstruction(points: np.ndarray) -> np.ndarray:
        return (points / SCALE) @ to_recon.T + offset

    background = np.random.default_rng(2).uniform([-900, -900, 1500], [900, 900, 1600], (40_000, 3))
    reconstruction = into_reconstruction(np.vstack([_surface(60_000, seed=1), background]))

    output_dir = root / "run"
    output_dir.mkdir()
    cameras = []
    for view, centre in enumerate(centres, start=1):
        pose = np.eye(4)
        pose[:3, 3] = into_reconstruction(centre[None])[0]
        cameras.append({"image": f"/data/scan1/rect_{view:03d}_3_r5000.png", "cam2world": pose})
    (output_dir / "cameras.json").write_text(
        json.dumps({"cameras": [{**c, "cam2world": c["cam2world"].tolist()} for c in cameras]})
    )
    return reference, reconstruction, output_dir


def test_view_number_reads_dtu_rectified_names() -> None:
    assert view_number("rect_017_3_r5000.png") == 17
    assert view_number(Path("/x/scan24/rect_049_0_r5000.png")) == 49
    assert view_number("frame_0001.png") is None


def _move_cameras(output_dir: Path, offsets_mm: dict[int, np.ndarray]) -> None:
    """Displace some estimated cameras by a DTU-frame distance, as a pose error would."""
    path = output_dir / "cameras.json"
    data = json.loads(path.read_text())
    for camera in data["cameras"]:
        offset = offsets_mm.get(view_number(camera["image"]))
        if offset is not None:
            pose = np.asarray(camera["cam2world"])
            pose[:3, 3] += np.asarray(offset) / SCALE
            camera["cam2world"] = pose.tolist()
    path.write_text(json.dumps(data))


def test_camera_fit_recovers_frame_and_scale(tmp_path: Path) -> None:
    reference, _, output_dir = _write_scan(tmp_path)
    fit = camera_alignment(load_reconstruction_cameras(output_dir), reference.camera_centres)
    assert fit.alignment.scale == pytest.approx(SCALE, rel=1e-9)
    assert fit.alignment.rmse < 1e-6
    assert fit.alignment.n_correspondences == N_VIEWS
    assert fit.spread > 0.05
    assert fit.outlier_views == []
    assert fit.pose_failure is False


def test_camera_fit_leaves_out_one_lost_camera(tmp_path: Path) -> None:
    reference, reconstruction, output_dir = _write_scan(tmp_path)
    _move_cameras(output_dir, {5: np.array([150.0, -80.0, 60.0])})

    fit = camera_alignment(load_reconstruction_cameras(output_dir), reference.camera_centres)
    assert fit.outlier_views == [5]
    # Untrimmed, the one view drags the fit past the failure line...
    assert fit.rmse_all_mm > 10.0
    # ...trimmed, the other eleven are exact and the rig has not failed.
    assert fit.alignment.rmse < 1e-6
    assert fit.pose_failure is False

    scores = score_dtu(reconstruction, reference, output_dir, tau=2.0)
    assert scores["camera_outliers"] == 1
    assert scores["camera_outlier_views"] == "5"
    assert scores["f_score"] > 0.99


def test_camera_fit_fails_a_rig_with_many_lost_cameras(tmp_path: Path) -> None:
    reference, _, output_dir = _write_scan(tmp_path)
    rng = np.random.default_rng(3)
    _move_cameras(output_dir, {view: rng.normal(0, 120.0, 3) for view in (2, 4, 7, 9)})

    fit = camera_alignment(load_reconstruction_cameras(output_dir), reference.camera_centres)
    # A third of the rig lost: trimmed or not, the fit must not pass as a good one.
    assert fit.alignment.rmse > 10.0 or fit.outlier_frac > 0.2
    assert fit.pose_failure is True


def test_camera_fit_fails_a_rig_that_drifted_everywhere(tmp_path: Path) -> None:
    """Pose error spread over every camera is not trimmed away: no view stands out."""
    reference, _, output_dir = _write_scan(tmp_path)
    rng = np.random.default_rng(4)
    _move_cameras(output_dir, {view: rng.normal(0, 15.0, 3) for view in range(1, N_VIEWS + 1)})

    fit = camera_alignment(load_reconstruction_cameras(output_dir), reference.camera_centres)
    assert fit.alignment.rmse > 10.0
    assert fit.pose_failure is True


def test_score_dtu_ignores_background_and_scale_error(tmp_path: Path) -> None:
    reference, reconstruction, output_dir = _write_scan(tmp_path)
    scores = score_dtu(reconstruction, reference, output_dir, tau=2.0)

    assert scores["camera_align_scale"] == pytest.approx(SCALE, rel=1e-6)
    # The background slab is outside the mask: only the 60k surface points count.
    assert scores["n_points_in_mask"] == 60_000
    assert scores["f_score"] > 0.99
    assert scores["dtu_overall"] < 1.0
    assert scores["align_method"] == "camera_sim3"
    assert scores["pose_failure"] is False
    assert scores["accuracy_outlier_frac"] == 0.0
    assert scores["f_score_icp"] > 0.99
    # The cameras were exact, so ICP has nothing to correct.
    assert scores["icp_shift_mm"] < 0.1
    assert scores["icp_rotation_deg"] < 0.05


def test_score_against_gt_takes_the_dtu_path_from_a_manifest(tmp_path: Path) -> None:
    reference, reconstruction, output_dir = _write_scan(tmp_path)
    (tmp_path / "images").mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "units": "mm",
                "scenes": [
                    {
                        "name": "scan1",
                        "image_dir": "images",
                        "gt_path": "stl001_total.ply",
                        "dtu_eval": {
                            "calibration_dir": "Calibration/cal18",
                            "obs_mask": "ObsMask1_10.mat",
                            "plane": "Plane1.mat",
                        },
                    }
                ],
            }
        )
    )
    scene = SceneSet.from_manifest(manifest).scenes[0]
    assert scene.dtu_reference() == reference

    scores = score_against_gt(
        reconstruction,
        np.empty((0, 3)),
        tau=2.0,
        reconstruction_scale=1000.0,
        reference=scene.dtu_reference(),
        output_dir=output_dir,
    )
    assert scores["protocol"] == "dtu_obsmask"
    assert scores["mast3r_scale_ratio"] == pytest.approx(SCALE / 1000.0, rel=1e-6)
    assert scores["f_score"] > 0.99


def test_estimated_normals_of_a_sphere_are_radial() -> None:
    points = np.random.default_rng(0).normal(size=(20_000, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    normals = estimate_normals(points)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    assert np.abs((normals * points).sum(axis=1)).mean() > 0.99


def _ellipsoid_in_reconstruction_frame(
    reference: DtuReference, output_dir: Path
) -> trimesh.Trimesh:
    """The scan's ellipsoid as a mesh, in the reconstruction's own frame."""
    fit = camera_alignment(load_reconstruction_cameras(output_dir), reference.camera_centres)
    mesh = trimesh.creation.icosphere(subdivisions=4)
    mesh.apply_scale([100.0, 70.0, 45.0])
    mesh.apply_translation([0, 0, 600.0])
    alignment = fit.alignment
    mesh.vertices = ((np.asarray(mesh.vertices) - alignment.translation) @ alignment.rotation) / (
        alignment.scale
    )
    return mesh


def test_mesh_normal_consistency_is_high_for_the_right_surface_and_lower_for_a_turned_one(
    tmp_path: Path,
) -> None:
    reference, _, output_dir = _write_scan(tmp_path)
    mesh = _ellipsoid_in_reconstruction_frame(reference, output_dir)

    right = mesh_normal_consistency(mesh, reference, output_dir)
    assert right["normal_consistency"] > 0.95
    assert right["normal_consistency_n"] > 1000

    # The same shape turned a quarter-turn about one axis through its centre no
    # longer lies on the ground truth, so what stays within the distance cap
    # meets normals that disagree.
    turned = mesh.copy()
    centre = turned.vertices.mean(axis=0)
    turn = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    turned.vertices = (turned.vertices - centre) @ turn.T + centre
    wrong = mesh_normal_consistency(turned, reference, output_dir)
    assert wrong["normal_consistency"] < right["normal_consistency"] - 0.1
