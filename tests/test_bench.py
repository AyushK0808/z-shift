"""Tests for the benchmark harness and the pipeline instrumentation it needs.

The harness produces the paper's numbers, so it has to be correct in the same
way the pipeline does: a fixture that does not inject the damage it claims, or
a metric that silently returns the wrong units, would be worse than no
experiment at all.

The experiment smoke tests run each Tier A module at `quick=True` against a
tmp_path results dir. They assert shape and invariants, not values, so they
stay fast and do not pin numbers that legitimately move with the code.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from bench.csvio import ResultWriter, aggregate, mean_std, read_rows
from bench.fixtures import (
    FACES_PER_HOLE,
    corrupt_mesh,
    derive_offset_stream,
    frames_from_video,
    make_motion_video,
)
from bench.gt_align import align_to_reference, apply_alignment, umeyama
from bench.instrument import cpu_model, env_metadata, peak_rss_mb, total_ram_gb
from bench.meshes import (
    analytic_colors,
    base_mesh,
    fragmented_mesh,
    ladder_mesh,
    mesh_at_triangle_count,
    to_pyvista,
)
from bench.metrics import (
    bbox_diagonal,
    chamfer_l1,
    hausdorff_95,
    normal_consistency,
    precision_recall_f,
    sample_points,
    unit_normalize,
    volume_ratio,
)
from spatial_ingestion.instrumentation import StageLog, current_rss_bytes, peak_rss_bytes
from spatial_ingestion.metadata.schema import FrameReference

# ---------------------------------------------------------------------------
# P1 - instrumentation
# ---------------------------------------------------------------------------


def test_stage_log_records_timing_and_memory() -> None:
    log = StageLog()
    with log.stage("touch", n_tri=123):
        buffer = np.ones((512, 512))
        buffer += 1

    assert len(log.stages) == 1
    record = log.stages[0]
    assert record["stage"] == "touch"
    assert record["n_tri"] == 123
    assert record["seconds"] >= 0.0
    assert {"peak_rss_mb", "rss_delta_mb", "current_rss_mb"} <= set(record)


def test_stage_log_records_a_stage_that_raises() -> None:
    """A step that fails still burned time; losing that row would hide the cost."""
    log = StageLog()
    with pytest.raises(ValueError, match="boom"), log.stage("explodes"):
        raise ValueError("boom")

    assert len(log.stages) == 1
    assert log.stages[0]["stage"] == "explodes"
    assert "ValueError: boom" in log.stages[0]["error"]


def test_stage_log_totals_and_dump(tmp_path: Path) -> None:
    log = StageLog()
    with log.stage("a"):
        pass
    with log.stage("b"):
        pass
    with log.stage("a"):
        pass

    assert log.total_seconds == pytest.approx(sum(s["seconds"] for s in log.stages), abs=1e-6)
    assert set(log.by_stage()) == {"a", "b"}

    path = log.dump(tmp_path / "nested" / "stages.json")
    assert json.loads(path.read_text(encoding="utf-8")) == log.as_list()


def test_peak_rss_is_available_on_this_platform() -> None:
    """Guards the cross-platform fallback: `resource` does not exist on Windows.

    A silent 0 would turn every memory column in the results into a lie, so this
    asserts a real number rather than tolerating the fallback.
    """
    assert peak_rss_bytes() > 0
    assert current_rss_bytes() > 0
    assert peak_rss_mb() > 0


def test_peak_rss_grows_with_touched_memory() -> None:
    before = peak_rss_bytes()
    ballast = np.ones((1024, 1024, 8))  # 64 MiB, touched by ones()
    ballast += 1
    assert peak_rss_bytes() >= before
    del ballast


def test_env_metadata_carries_what_the_setup_subsection_must_state() -> None:
    meta = env_metadata()
    for key in ("cpu", "os", "python", "numpy", "trimesh", "pyvista"):
        assert meta[key], f"{key} missing from env metadata"
    assert isinstance(meta["cuda_available"], bool)
    assert cpu_model()
    ram_gb = total_ram_gb()
    assert ram_gb is None or ram_gb > 0


# ---------------------------------------------------------------------------
# clean_mesh instrumentation (P1 wiring)
# ---------------------------------------------------------------------------


def test_clean_mesh_emits_stage_timings() -> None:
    from spatial_ingestion.refinement import MeshCleaningConfig, clean_mesh

    mesh = to_pyvista(ladder_mesh("icosphere", 5_000))
    result = clean_mesh(mesh, MeshCleaningConfig(mode="object", smoothing_iters=5))

    stages = [record["stage"] for record in result["stage_timings"]]
    assert stages[:4] == ["component_filter", "fill_holes", "smooth", "finalize"]
    assert "watertight_check" in stages
    assert result["total_stage_seconds"] == pytest.approx(
        sum(r["seconds"] for r in result["stage_timings"]), abs=1e-6
    )


def test_refinement_diagnostics_stay_json_serialisable() -> None:
    """Stage timings ride into refinement_manifest.json via _serializable_diagnostics."""
    from spatial_ingestion.final_pipeline.core import _serializable_diagnostics
    from spatial_ingestion.refinement import MeshCleaningConfig, clean_mesh

    result = clean_mesh(
        to_pyvista(ladder_mesh("icosphere", 2_000)),
        MeshCleaningConfig(mode="object", smoothing_iters=0),
    )
    payload = json.dumps(_serializable_diagnostics(result))
    assert "stage_timings" in json.loads(payload)


def test_build_run_manifest_records_output_side_fields() -> None:
    from spatial_ingestion.reconstruction.export import build_run_manifest

    manifest = build_run_manifest(
        image_paths=[Path("a.jpg"), Path("b.jpg")],
        output_dir=Path("out"),
        output_path=Path("out/mesh.glb"),
        model_name="naver/model",
        device="cpu",
        image_size=512,
        pairing_strategy="swin",
        dry_run=False,
        outputs={"n_pairs": 12, "n_vertices": 900},
    )
    assert manifest["outputs"]["n_frames_used"] == 2
    assert manifest["outputs"]["n_pairs"] == 12
    assert manifest["outputs"]["n_vertices"] == 900
    assert "deterministic_algorithms" in manifest["reproducibility"]


def test_set_seed_records_the_determinism_request() -> None:
    from spatial_ingestion.reconstruction.device import reproducibility_metadata, set_seed

    set_seed(0)
    assert reproducibility_metadata()["deterministic_algorithms"] is True


# ---------------------------------------------------------------------------
# P2 - metrics
# ---------------------------------------------------------------------------


def test_unit_normalize_gives_a_unit_diagonal_centred_at_origin() -> None:
    mesh = unit_normalize(trimesh.creation.box(extents=(3.0, 5.0, 7.0)))
    assert bbox_diagonal(mesh) == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(mesh.bounds.mean(axis=0), 0.0, atol=1e-9)


def test_chamfer_of_a_mesh_against_itself_is_near_zero() -> None:
    mesh = unit_normalize(trimesh.creation.icosphere(subdivisions=4))
    points_a = sample_points(mesh, 20_000, seed=0)
    points_b = sample_points(mesh, 20_000, seed=1)
    assert chamfer_l1(points_a, points_b) < 0.01


def test_chamfer_tracks_a_known_offset() -> None:
    """Two parallel planes offset by d must be exactly d apart. Pins the units.

    A translated *sphere* would not work: its shifted samples still lie on a
    sphere, so nearest neighbours are far closer than the translation.
    """
    rng = np.random.default_rng(0)
    plane = np.column_stack(
        [rng.uniform(-1, 1, 20_000), rng.uniform(-1, 1, 20_000), np.zeros(20_000)]
    )
    offset = 0.05
    shifted = plane + np.array([0.0, 0.0, offset])
    assert chamfer_l1(plane, shifted) == pytest.approx(offset, rel=0.05)


def test_hausdorff_95_is_at_least_chamfer() -> None:
    mesh = unit_normalize(trimesh.creation.icosphere(subdivisions=3))
    points = sample_points(mesh, 10_000, seed=0)
    noisy = points + np.random.default_rng(0).normal(0, 0.01, points.shape)
    assert hausdorff_95(points, noisy) >= chamfer_l1(points, noisy)


def test_precision_recall_f_is_perfect_for_identical_sets() -> None:
    points = sample_points(unit_normalize(trimesh.creation.icosphere(3)), 5_000, seed=0)
    precision, recall, f_score = precision_recall_f(points, points, tau=0.01)
    assert (precision, recall, f_score) == (1.0, 1.0, 1.0)


def test_precision_recall_f_is_zero_for_disjoint_sets() -> None:
    points = sample_points(unit_normalize(trimesh.creation.icosphere(3)), 2_000, seed=0)
    far_away = points + 100.0
    assert precision_recall_f(points, far_away, tau=0.01) == (0.0, 0.0, 0.0)


def test_normal_consistency_matches_the_analytic_sphere_reference() -> None:
    """Pins the KD-tree estimator against a closed-form answer.

    For a sphere the true surface normal is radial, so the mean |cos| between
    sampled facet normals and the radial direction is what the metric must
    return. `rtree` is not installed, so `exact=True` is unavailable and this
    is how the fast path is validated.
    """
    from bench.metrics import sample_points_with_normals

    sphere = unit_normalize(trimesh.creation.icosphere(subdivisions=4))
    points, facet_normals = sample_points_with_normals(sphere, 20_000, seed=0)
    radial = points / np.linalg.norm(points, axis=1, keepdims=True)
    analytic = float(np.abs((facet_normals * radial).sum(axis=1)).mean())

    measured = normal_consistency(sphere, sphere, n=20_000, seed=0)
    assert measured == pytest.approx(analytic, abs=0.005)
    assert measured > 0.99


def test_normal_consistency_drops_when_normals_are_perturbed() -> None:
    sphere = unit_normalize(trimesh.creation.icosphere(subdivisions=4))
    noisy = sphere.copy()
    noisy.vertices = np.asarray(noisy.vertices) + np.random.default_rng(0).normal(
        0, 0.004, np.asarray(noisy.vertices).shape
    )
    assert normal_consistency(noisy, sphere, n=20_000, seed=0) < normal_consistency(
        sphere, sphere, n=20_000, seed=0
    )


def test_normal_consistency_exact_reports_the_missing_dependency() -> None:
    sphere = unit_normalize(trimesh.creation.icosphere(subdivisions=2))
    try:
        import rtree  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="rtree"):
            normal_consistency(sphere, sphere, n=500, seed=0, exact=True)
    else:  # pragma: no cover - only when the optional dep is installed
        assert normal_consistency(sphere, sphere, n=500, seed=0, exact=True) > 0.9


def test_volume_ratio_is_none_for_open_surfaces() -> None:
    sheet = base_mesh("pointmap_sheet")
    assert not sheet.is_watertight
    assert volume_ratio(sheet, sheet) is None


def test_volume_ratio_detects_a_known_scale() -> None:
    sphere = trimesh.creation.icosphere(subdivisions=3)
    bigger = sphere.copy()
    bigger.apply_scale(2.0)
    assert volume_ratio(bigger, sphere) == pytest.approx(8.0, rel=1e-6)


# ---------------------------------------------------------------------------
# P3 - fixtures
# ---------------------------------------------------------------------------


def test_corrupt_mesh_noise_scales_with_the_bounding_diagonal() -> None:
    mesh = unit_normalize(trimesh.creation.icosphere(subdivisions=3))
    damaged = corrupt_mesh(mesh, noise_sigma=0.01, seed=0)
    displacement = np.linalg.norm(np.asarray(damaged.vertices) - np.asarray(mesh.vertices), axis=1)
    # Norm of a 3-vector of N(0, sigma) has mean ~ sigma * sqrt(8/pi).
    assert displacement.mean() == pytest.approx(0.01 * np.sqrt(8 / np.pi), rel=0.2)


def test_corrupt_mesh_removes_connected_patches_not_scattered_faces() -> None:
    """5 holes must mean 5 holes, not 100 single-triangle punctures."""
    mesh = ladder_mesh("icosphere", 20_000)
    damaged = corrupt_mesh(mesh, n_holes=5, seed=0)
    assert len(damaged.faces) == len(mesh.faces) - 5 * FACES_PER_HOLE
    assert len(damaged.split(only_watertight=False)) == 1


def test_corrupt_mesh_injects_the_requested_number_of_fragments() -> None:
    mesh = ladder_mesh("icosphere", 5_000)
    damaged = corrupt_mesh(mesh, n_fragments=5, seed=0)
    assert len(damaged.split(only_watertight=False)) == 6


def test_corrupt_mesh_is_deterministic_for_a_seed() -> None:
    mesh = ladder_mesh("icosphere", 5_000)
    kwargs = {"noise_sigma": 0.005, "n_holes": 3, "n_fragments": 2}
    first = corrupt_mesh(mesh, seed=7, **kwargs)
    second = corrupt_mesh(mesh, seed=7, **kwargs)
    assert np.array_equal(first.vertices, second.vertices)
    assert np.array_equal(first.faces, second.faces)


def test_make_motion_video_labels_its_own_schedule(tmp_path: Path) -> None:
    windows = make_motion_video(
        tmp_path / "clip.mp4", fps=12, seconds=4, motion_windows=((1.0, 2.0),)
    )
    assert windows == [(1000.0, 2000.0)]
    assert (tmp_path / "clip.mp4").stat().st_size > 0


def test_camera_pan_is_what_reaches_the_high_motion_branch(tmp_path: Path) -> None:
    """Without a panning backdrop the sampler's 0.18 threshold is unreachable.

    The moving subject's motion score is bounded by its own area, so this pins
    the reason the fixture pans the whole frame.
    """
    schedule = ((0.5, 5.5),)
    with_pan = tmp_path / "pan.mp4"
    without_pan = tmp_path / "nopan.mp4"
    make_motion_video(with_pan, fps=24, seconds=6, motion_windows=schedule, speed_px_per_frame=12)
    make_motion_video(
        without_pan,
        fps=24,
        seconds=6,
        motion_windows=schedule,
        speed_px_per_frame=12,
        camera_pan=False,
    )

    panned = max(f.motion_score or 0.0 for f in frames_from_video(with_pan))
    still = [f.motion_score or 0.0 for f in frames_from_video(without_pan)][1:]
    assert panned >= 0.18
    assert max(still) < 0.18


def test_frames_from_video_produces_ordered_timestamped_frames(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    make_motion_video(video, fps=24, seconds=5, motion_windows=((1.0, 4.0),))
    frames = frames_from_video(video, source_id="cam_x")

    assert len(frames) >= 2
    assert all(frame.source_id == "cam_x" for frame in frames)
    assert [f.index for f in frames] == sorted(f.index for f in frames)
    assert all(f.timestamp_ms is not None and f.uri for f in frames)


def test_derive_offset_stream_injects_exactly_the_requested_offset() -> None:
    source = [
        FrameReference(frame_id=f"f{i}", index=i, timestamp_ms=i * 100.0, motion_score=0.3)
        for i in range(5)
    ]
    derived = derive_offset_stream(source, 250.0, source_id="cam_b")
    assert [f.timestamp_ms for f in derived] == [250.0, 350.0, 450.0, 550.0, 650.0]
    assert all(f.source_id == "cam_b" for f in derived)
    # The source list must not be mutated; the syncer is handed both.
    assert [f.timestamp_ms for f in source] == [0.0, 100.0, 200.0, 300.0, 400.0]


def test_derive_offset_stream_motion_noise_perturbs_scores() -> None:
    source = [
        FrameReference(frame_id=f"f{i}", index=i, timestamp_ms=i * 100.0, motion_score=0.3)
        for i in range(20)
    ]
    identical = derive_offset_stream(source, 0.0, motion_noise=0.0)
    perturbed = derive_offset_stream(source, 0.0, motion_noise=0.05, seed=1)
    assert [f.motion_score for f in identical] == [0.3] * 20
    assert [f.motion_score for f in perturbed] != [0.3] * 20
    assert all(0.0 <= (f.motion_score or 0.0) <= 1.0 for f in perturbed)


# ---------------------------------------------------------------------------
# mesh ladder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_tri", [1_000, 5_000, 20_000])
def test_ladder_hits_its_triangle_target(n_tri: int) -> None:
    mesh = ladder_mesh("icosphere", n_tri)
    assert abs(len(mesh.faces) - n_tri) <= max(10, n_tri * 0.02)


def test_the_two_ladder_sources_differ_where_it_matters() -> None:
    """The sheet source exists to exercise the is_sheet_like hole-fill guard."""
    from spatial_ingestion.refinement.core import is_sheet_like

    sphere = ladder_mesh("icosphere", 5_000)
    sheet = ladder_mesh("pointmap_sheet", 5_000)
    assert sphere.is_watertight
    assert not sheet.is_watertight
    assert not is_sheet_like(to_pyvista(sphere))
    assert is_sheet_like(to_pyvista(sheet))


def test_fragmented_mesh_has_the_requested_component_count() -> None:
    mesh = fragmented_mesh(4_000, 8, seed=0)
    assert len(mesh.split(only_watertight=False)) == 8


def test_mesh_at_triangle_count_can_subdivide_upwards() -> None:
    small = trimesh.creation.icosphere(subdivisions=1)
    grown = mesh_at_triangle_count(small, 4_000)
    assert len(grown.faces) > len(small.faces)


def test_analytic_colors_are_a_closed_form_function_of_position() -> None:
    mesh = unit_normalize(trimesh.creation.box())
    colors = analytic_colors(mesh.vertices)
    assert colors.dtype == np.uint8
    assert colors.shape == (len(mesh.vertices), 3)
    # Corner vertices map to the extremes of every channel.
    assert colors.min() == 0
    assert colors.max() == 255


def test_to_pyvista_round_trips_face_count() -> None:
    mesh = ladder_mesh("icosphere", 2_000)
    assert to_pyvista(mesh).n_cells == len(mesh.faces)


# ---------------------------------------------------------------------------
# CSV plumbing
# ---------------------------------------------------------------------------


def test_result_writer_unions_columns_across_rows(tmp_path: Path) -> None:
    writer = ResultWriter("demo", tmp_path)
    writer.add(a=1)
    writer.add(a=2, b="only-on-this-row")
    path = writer.write()

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {"a", "b"} <= set(rows[0])
    assert rows[0]["b"] == ""
    assert rows[1]["b"] == "only-on-this-row"


def test_result_writer_stamps_the_environment_on_every_row(tmp_path: Path) -> None:
    """A number separated from its machine is not a measurement."""
    writer = ResultWriter("demo", tmp_path)
    writer.add(value=1)
    rows = read_rows(writer.write())
    assert rows[0]["env_cpu"]
    assert rows[0]["env_python"]
    assert rows[0]["env_run_started_utc"]


def test_read_rows_coerces_numbers_and_booleans(tmp_path: Path) -> None:
    writer = ResultWriter("demo", tmp_path)
    writer.add(count=3, ratio=0.5, flag=True, label="text", blank="")
    writer.write()
    row = read_rows("demo", tmp_path)[0]
    assert row["count"] == 3 and isinstance(row["count"], int)
    assert row["ratio"] == 0.5
    assert row["flag"] is True
    assert row["label"] == "text"
    assert row["blank"] is None


def test_mean_std_uses_the_sample_standard_deviation() -> None:
    mean, std, n = mean_std([1.0, 2.0, 3.0])
    assert (mean, n) == (2.0, 3)
    assert std == pytest.approx(1.0)  # ddof=1, not 0.816 (ddof=0)


def test_mean_std_handles_empty_and_singleton_inputs() -> None:
    assert mean_std([])[2] == 0
    assert mean_std([5.0]) == (5.0, 0.0, 1)


def test_aggregate_groups_and_reduces() -> None:
    rows = [
        {"k": "a", "v": 1.0},
        {"k": "a", "v": 3.0},
        {"k": "b", "v": 10.0},
        {"k": "b", "v": None},
    ]
    result = aggregate(rows, ["k"], "v")
    assert result[("a",)][0] == 2.0
    assert result[("b",)] == (10.0, 0.0, 1)


# ---------------------------------------------------------------------------
# Tier B alignment (runs on CPU; the reconstruction it serves does not)
# ---------------------------------------------------------------------------


def test_umeyama_recovers_a_known_rigid_transform() -> None:
    rng = np.random.default_rng(0)
    source = rng.normal(size=(50, 3))
    angle = np.deg2rad(23.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([1.5, -2.0, 0.25])
    target = source @ rotation.T + translation

    result = umeyama(source, target)
    assert result.rmse < 1e-9
    assert np.allclose(result.rotation, rotation, atol=1e-8)
    assert np.allclose(result.translation, translation, atol=1e-8)
    assert np.allclose(apply_alignment(source, result), target, atol=1e-8)


def test_umeyama_recovers_scale_only_when_asked() -> None:
    rng = np.random.default_rng(1)
    source = rng.normal(size=(60, 3))
    target = 3.0 * source

    assert umeyama(source, target, with_scale=True).scale == pytest.approx(3.0, rel=1e-6)
    assert umeyama(source, target, with_scale=False).scale == 1.0


def test_umeyama_does_not_fit_a_reflection() -> None:
    rng = np.random.default_rng(2)
    source = rng.normal(size=(40, 3))
    mirrored = source * np.array([1.0, 1.0, -1.0])
    result = umeyama(source, mirrored)
    assert np.linalg.det(result.rotation) == pytest.approx(1.0, abs=1e-8)


def test_align_to_reference_reduces_the_residual() -> None:
    rng = np.random.default_rng(3)
    target = rng.normal(size=(400, 3))
    source = target + np.array([0.4, -0.3, 0.2])
    aligned, result = align_to_reference(source, target)
    assert result.rmse < np.linalg.norm(source - target, axis=1).mean()
    assert aligned.shape == source.shape
    assert result.as_row()["align_with_scale"] is False


def test_umeyama_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        umeyama(np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="matching shapes"):
        umeyama(np.zeros((5, 3)), np.zeros((4, 3)))


# ---------------------------------------------------------------------------
# Tier A experiment smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_name",
    [
        "bench.exp_a1_refine_scaling",
        "bench.exp_a3_refine_quality",
        "bench.exp_a4_color_transfer",
        "bench.exp_a5_frame_budget",
        "bench.exp_a6_sync_offset",
        "bench.exp_a7_motion_sampling",
        "bench.exp_a8_rigging_quality",
        "bench.exp_a9_routing_matrix",
    ],
)
def test_tier_a_experiment_writes_a_csv(module_name: str, tmp_path: Path) -> None:
    import importlib

    module = importlib.import_module(module_name)
    writer = module.run(results_dir=tmp_path, quick=True, seed=0)
    path = writer.write()

    assert len(writer) > 0
    assert path.exists()
    rows = read_rows(path)
    assert len(rows) == len(writer)
    assert all(row["exp_id"] == module.EXP_ID for row in rows)


def test_a2_stage_profile_runs_without_the_profiler(tmp_path: Path) -> None:
    """cProfile is skipped here: it is slow and adds no coverage of the logic."""
    from bench import exp_a2_stage_profile

    writer = exp_a2_stage_profile.run(results_dir=tmp_path, quick=True, seed=0, profile=False)
    assert len(writer) > 0
    assert {"component_filter", "finalize"} <= {row["stage"] for row in writer.rows}


# ---------------------------------------------------------------------------
# Findings the experiments exist to detect. These pin *current behaviour*, so
# changing the behaviour deliberately will fail them and force the paper text
# to be updated at the same time.
# ---------------------------------------------------------------------------


def test_object_mode_keeps_every_fragment_it_is_given() -> None:
    """A3's expected negative result, pinned.

    SIV-D/SV-B say component filtering "removes isolated fragments". In object
    mode `keep_object_components` merges every piece and drops none. If this
    test starts failing, the code was changed to match the paper -- update
    A3's commentary and the paper text together.
    """
    from spatial_ingestion.refinement import MeshCleaningConfig, clean_mesh

    mesh = corrupt_mesh(ladder_mesh("icosphere", 5_000), n_fragments=5, seed=0)
    assert len(mesh.split(only_watertight=False)) == 6

    result = clean_mesh(to_pyvista(mesh), MeshCleaningConfig(mode="object", smoothing_iters=0))
    assert len(result["mesh"].split_bodies()) == 6


def test_hole_filling_is_skipped_on_sheet_like_input() -> None:
    """The other expected negative: `is_sheet_like` gates hole filling.

    Fused per-view pointmaps are sheet-like, so this branch is the one real
    reconstruction output takes.
    """
    from spatial_ingestion.refinement.core import fill_mesh_holes, is_sheet_like

    sheet = to_pyvista(ladder_mesh("pointmap_sheet", 5_000)).extract_surface(algorithm=None)
    assert is_sheet_like(sheet)
    assert fill_mesh_holes(sheet, hole_size=None) is sheet


def test_cap_frames_does_not_restore_capture_order() -> None:
    """A5's finding, pinned at the unit level.

    Above the budget, `_cap_frames` returns motion-rank order. `swin` pairing
    pairs by list adjacency, so this ordering reaches pairing directly.
    """
    from spatial_ingestion.config import MAX_RECONSTRUCTION_FRAMES
    from spatial_ingestion.reconstruction.jobs import ReconstructionJobBuilder
    from spatial_ingestion.reconstruction.models import HandoffFrame

    over_budget = MAX_RECONSTRUCTION_FRAMES + 10
    # 37 is coprime with over_budget, so this is a fixed scrambling permutation
    # of the motion ranks. A monotonic score would make motion-rank order and
    # capture order identical and hide the very thing being tested.
    frames = [
        HandoffFrame(
            frame_id=f"f{i}",
            uri=f"file:///tmp/f{i}.jpg",
            index=i,
            motion_score=((i * 37) % over_budget) / over_budget,
        )
        for i in range(over_budget)
    ]
    capped = ReconstructionJobBuilder._cap_frames(frames)

    assert len(capped) == MAX_RECONSTRUCTION_FRAMES
    indices = [frame.index for frame in capped]
    assert indices != sorted(indices), (
        "capped frames came back in capture order; if _cap_frames was changed "
        "to re-sort, update A5's commentary and the Section IV-C text too"
    )


# ---------------------------------------------------------------------------
# A9 as a regression test, per the protocol
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source_type", "use_case", "expected"),
    [
        ("single_image", "editing", "deliverable"),
        ("single_image", "viewing", "InvalidRoutingError"),
        ("single_image", "live", "InvalidRoutingError"),
        ("image_folder", "editing", "deliverable"),
        ("image_folder", "viewing", "deliverable"),
        ("image_folder", "live", "InvalidRoutingError"),
        ("single_video", "editing", "deliverable"),
        ("single_video", "viewing", "deliverable"),
        ("single_video", "live", "InvalidRoutingError"),
        ("video_folder", "editing", "deliverable"),
        ("video_folder", "viewing", "deliverable"),
        ("video_folder", "live", "InvalidRoutingError"),
        ("live_stream", "editing", "InvalidRoutingError"),
        ("live_stream", "viewing", "InvalidRoutingError"),
        ("live_stream", "live", "TrackNotImplementedError"),
        ("unknown", "editing", "InvalidRoutingError"),
        ("unknown", "viewing", "InvalidRoutingError"),
        ("unknown", "live", "InvalidRoutingError"),
        ("image_folder", "not_a_use_case", "InvalidRoutingError"),
        ("live_stream", "not_a_use_case", "InvalidRoutingError"),
    ],
)
def test_routing_matrix_has_no_silent_fallthrough(
    source_type: str, use_case: str, expected: str, tmp_path: Path
) -> None:
    """SIV-E: every combination either routes or raises explicitly."""
    from spatial_ingestion.metadata.schema import SourceType
    from spatial_ingestion.outcomes_engine.engine import (
        InvalidRoutingError,
        TrackNotImplementedError,
        deliverable_router,
    )

    try:
        result = deliverable_router(SourceType(source_type), use_case, output_root=tmp_path)
        outcome = "deliverable"
        assert result.output_path and Path(result.output_path).exists()
    except InvalidRoutingError:
        outcome = "InvalidRoutingError"
    except TrackNotImplementedError:
        outcome = "TrackNotImplementedError"

    assert outcome == expected


# ---------------------------------------------------------------------------
# Tier B plumbing that does not need MASt3R or a GPU.
#
# The reconstruct-and-score loops cannot run here, but the parts that would
# silently mis-handle a dataset -- manifest parsing, ground-truth loading,
# alignment-cache clearing -- can and should be.
# ---------------------------------------------------------------------------


def _write_manifest(tmp_path: Path, **overrides: object) -> Path:
    images = tmp_path / "scene_a" / "images"
    images.mkdir(parents=True)
    for index in range(3):
        (images / f"{index:03d}.jpg").write_bytes(b"not-a-real-jpeg")
    gt = tmp_path / "scene_a" / "gt.ply"
    trimesh.PointCloud(np.random.default_rng(0).normal(size=(500, 3))).export(str(gt))

    payload = {
        "tau": 2.0,
        "units": "mm",
        "scenes": [
            {"name": "scene_a", "image_dir": "scene_a/images", "gt_path": "scene_a/gt.ply"},
            {"name": "scene_b", "image_dir": "scene_a/images", "tau": 0.01, "units": "scene"},
        ],
        **overrides,
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def test_scene_manifest_resolves_paths_relative_to_itself(tmp_path: Path) -> None:
    from bench.tier_b_common import SceneSet, iter_scenes

    scene_set = SceneSet.from_manifest(_write_manifest(tmp_path))
    assert [scene.name for scene in scene_set.scenes] == ["scene_a", "scene_b"]

    first = scene_set.scenes[0]
    assert first.image_dir.is_absolute() and first.image_dir.exists()
    assert first.gt_path is not None and first.gt_path.exists()
    assert first.tau == 2.0 and first.units == "mm"

    # A per-scene tau must override the manifest default; an F-score reported
    # against the wrong tau is worse than none.
    assert scene_set.scenes[1].tau == 0.01
    assert scene_set.scenes[1].gt_path is None

    assert [s.name for s in iter_scenes(scene_set, ["scene_b"])] == ["scene_b"]
    assert len(iter_scenes(scene_set, None)) == 2


def test_scene_image_paths_are_sorted_and_limited(tmp_path: Path) -> None:
    from bench.tier_b_common import SceneSet

    scene = SceneSet.from_manifest(_write_manifest(tmp_path)).scenes[0]
    paths = scene.image_paths()
    assert [p.name for p in paths] == ["000.jpg", "001.jpg", "002.jpg"]
    assert len(scene.image_paths(2)) == 2


def test_image_paths_order_frames_numerically_not_lexically(tmp_path: Path) -> None:
    # NeRF-synthetic numbers frames without padding, so a lexical sort puts
    # r_109 before r_11. Every B module reads this order as capture order:
    # B3 pairs within a window of it and B4 samples a budget from it.
    from bench.tier_b_common import Scene

    images = tmp_path / "unpadded"
    images.mkdir()
    for index in (0, 1, 2, 9, 10, 11, 100, 109, 110):
        (images / f"r_{index}.png").write_bytes(b"not-a-real-png")

    paths = Scene(name="x", image_dir=images).image_paths()
    assert [p.name for p in paths] == [
        "r_0.png",
        "r_1.png",
        "r_2.png",
        "r_9.png",
        "r_10.png",
        "r_11.png",
        "r_100.png",
        "r_109.png",
        "r_110.png",
    ]


def test_image_paths_exclude_depth_and_normal_maps(tmp_path: Path) -> None:
    # A .png depth map is indistinguishable from a photograph by suffix, and
    # MASt3R will happily consume one and produce a worse reconstruction
    # without raising anything.
    from bench.tier_b_common import Scene

    images = tmp_path / "nerf"
    images.mkdir()
    for index in range(3):
        (images / f"r_{index}.png").write_bytes(b"not-a-real-png")
        (images / f"r_{index}_depth_0000.png").write_bytes(b"not-a-real-png")
        (images / f"r_{index}_normal_0000.png").write_bytes(b"not-a-real-png")

    paths = Scene(name="x", image_dir=images).image_paths()
    assert [p.name for p in paths] == ["r_0.png", "r_1.png", "r_2.png"]


def test_image_paths_keep_names_that_merely_contain_a_map_word(tmp_path: Path) -> None:
    # DTU's rect_001_0_r5000.png and a file called depthscan_01.png are
    # photographs; the exclusion must key on a delimited word, not a substring.
    from bench.tier_b_common import Scene

    images = tmp_path / "dtu"
    images.mkdir()
    for name in ("rect_001_0_r5000.png", "rect_002_0_r5000.png", "depthscan_01.png"):
        (images / name).write_bytes(b"not-a-real-png")

    paths = Scene(name="x", image_dir=images).image_paths()
    assert [p.name for p in paths] == [
        "depthscan_01.png",
        "rect_001_0_r5000.png",
        "rect_002_0_r5000.png",
    ]


def test_image_glob_selects_one_dtu_lighting_condition(tmp_path: Path) -> None:
    # DTU photographs each viewpoint under 7 lighting conditions. Without a
    # glob the scene hands MASt3R seven near-duplicate copies of every pose:
    # the pair count multiplies and no new parallax arrives.
    from bench.tier_b_common import Scene

    images = tmp_path / "scan24"
    images.mkdir()
    for view in range(1, 4):
        for light in range(7):
            (images / f"rect_{view:03d}_{light}_r5000.png").write_bytes(b"not-a-real-png")

    scene = Scene(name="scan24", image_dir=images)
    assert len(scene.image_paths()) == 21

    filtered = Scene(name="scan24", image_dir=images, image_glob="*_3_r5000.png")
    assert [p.name for p in filtered.image_paths()] == [
        "rect_001_3_r5000.png",
        "rect_002_3_r5000.png",
        "rect_003_3_r5000.png",
    ]


def test_image_glob_that_matches_nothing_fails_loudly(tmp_path: Path) -> None:
    from bench.tier_b_common import Scene

    images = tmp_path / "scan24"
    images.mkdir()
    (images / "rect_001_0_r5000.png").write_bytes(b"not-a-real-png")

    scene = Scene(name="scan24", image_dir=images, image_glob="*_9_r5000.png")
    with pytest.raises(FileNotFoundError, match="matching"):
        scene.image_paths()


def test_manifest_carries_image_glob(tmp_path: Path) -> None:
    from bench.tier_b_common import SceneSet

    images = tmp_path / "scan24"
    images.mkdir()
    for light in range(3):
        (images / f"rect_001_{light}_r5000.png").write_bytes(b"not-a-real-png")
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {
                "scenes": [
                    {
                        "name": "scan24",
                        "image_dir": "scan24",
                        "image_glob": "*_1_r5000.png",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    scene = SceneSet.from_manifest(manifest).scenes[0]
    assert scene.image_glob == "*_1_r5000.png"
    assert [p.name for p in scene.image_paths()] == ["rect_001_1_r5000.png"]


def test_scene_without_images_fails_loudly(tmp_path: Path) -> None:
    from bench.tier_b_common import Scene

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no images"):
        Scene(name="x", image_dir=empty).image_paths()


def test_load_gt_points_handles_point_clouds_and_meshes(tmp_path: Path) -> None:
    from bench.tier_b_common import load_gt_points

    cloud_path = tmp_path / "cloud.ply"
    trimesh.PointCloud(np.random.default_rng(0).normal(size=(2_000, 3))).export(str(cloud_path))
    cloud = load_gt_points(cloud_path, max_points=500, seed=0)
    assert cloud.shape == (500, 3)

    mesh_path = tmp_path / "mesh.ply"
    trimesh.creation.icosphere(subdivisions=3).export(str(mesh_path))
    sampled = load_gt_points(mesh_path, max_points=1_000, seed=0)
    assert sampled.shape == (1_000, 3)


def test_clear_alignment_cache_removes_the_cache_and_tolerates_absence(tmp_path: Path) -> None:
    """Without this, a repeat run replays the cached alignment and B7 is vacuous."""
    from bench.tier_b_common import clear_alignment_cache

    cache = tmp_path / "cache"
    (cache / "nested").mkdir(parents=True)
    (cache / "nested" / "blob.bin").write_bytes(b"x")

    clear_alignment_cache(tmp_path)
    assert not cache.exists()
    clear_alignment_cache(tmp_path)  # idempotent


def test_build_job_carries_params_and_uris(tmp_path: Path) -> None:
    from bench.tier_b_common import build_job
    from spatial_ingestion.reconstruction._io import uri_to_path
    from spatial_ingestion.reconstruction.models import Mast3rRunParams

    images = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    for path in images:
        path.write_bytes(b"x")

    job = build_job(
        images,
        tmp_path / "out" / "mesh.glb",
        params=Mast3rRunParams(image_size=224, pairing_strategy="swin", seed=7),
        label="unit",
    )
    assert [uri_to_path(uri).name for uri in job.image_uris] == ["a.jpg", "b.jpg"]
    assert job.metadata["image_size"] == 224
    assert job.metadata["pairing_strategy"] == "swin"
    assert job.metadata["seed"] == 7
    assert job.label == "unit"


def test_set_address_space_limit_reports_unsupported_on_windows() -> None:
    """B5 must not report a fallback rate of zero that really means 'never tested'."""
    import sys

    from bench.exp_b5_tsdf_fallback import set_address_space_limit

    if sys.platform == "win32":
        assert set_address_space_limit(4.0) is False
    else:  # pragma: no cover - depends on the runner's platform
        assert isinstance(set_address_space_limit(1024.0), bool)


def test_b4_builds_frames_with_motion_scores(tmp_path: Path) -> None:
    import cv2

    from bench.exp_b4_frame_budget_ablation import _frames_from_image_dir

    paths = []
    for index in range(4):
        path = tmp_path / f"{index:03d}.png"
        frame = np.full((64, 64, 3), index * 60 % 255, np.uint8)
        cv2.imwrite(str(path), frame)
        paths.append(path)

    frames = _frames_from_image_dir(paths)
    assert [f.index for f in frames] == [0, 1, 2, 3]
    assert frames[0].motion_score == 1.0
    assert all(0.0 <= (f.motion_score or 0.0) <= 1.0 for f in frames)
    assert all(f.timestamp_ms is not None for f in frames)
