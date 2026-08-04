from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from spatial_ingestion.final_pipeline.cli import main as final_pipeline_cli_main
from spatial_ingestion.final_pipeline.core import (
    PipelineArtifactError,
    run_full_pipeline,
    run_phase2_phase3_pipeline,
)
from spatial_ingestion.reconstruction.cli import collect_input_images
from spatial_ingestion.reconstruction.models import (
    Mast3rRunParams,
    ReconstructionJob,
    ReconstructionMode,
)
from spatial_ingestion.refinement import MeshCleaningConfig
from spatial_ingestion.test_harness.media_factory import create_sample_image


def _job(output_path: Path) -> ReconstructionJob:
    return ReconstructionJob(
        mode=ReconstructionMode.MULTI_VIEW,
        image_uris=["image1.jpg", "image2.jpg"],
        output_path=str(output_path),
    )


def test_pipeline_runs_reconstruction_then_refinement(fake_reconstruction, tmp_path: Path) -> None:
    raw_mesh_path = tmp_path / "mesh.obj"
    fake_reconstruction(raw_mesh_path=raw_mesh_path)

    result = run_phase2_phase3_pipeline(
        _job(raw_mesh_path),
        MeshCleaningConfig(smoothing_iters=0, verify_watertight=False),
    )

    assert result.raw_mesh_path == raw_mesh_path.resolve()
    assert result.refined_mesh_path.exists()
    assert result.refinement_manifest_path.exists()
    assert result.refinement_diagnostics["output_point_count"] > 0


def test_pipeline_fails_when_phase2_mesh_is_missing(fake_reconstruction, tmp_path: Path) -> None:
    raw_mesh_path = tmp_path / "missing.obj"
    fake_reconstruction(raw_mesh_path=raw_mesh_path, write_mesh=False)

    try:
        run_phase2_phase3_pipeline(_job(raw_mesh_path), MeshCleaningConfig(smoothing_iters=0))
    except PipelineArtifactError as exc:
        assert "no mesh artifact" in str(exc)
    else:
        raise AssertionError("expected missing Phase 2 mesh to fail")


def test_pipeline_cli_smoke_with_mocked_phase2(fake_reconstruction, tmp_path: Path, capsys) -> None:
    image_dir = tmp_path / "views"
    image_dir.mkdir()
    create_sample_image(image_dir / "front.png")
    create_sample_image(image_dir / "side.png")
    raw_mesh_path = tmp_path / "pipeline_out" / "mesh.obj"

    fake_reconstruction(raw_mesh_path=raw_mesh_path)

    assert (
        final_pipeline_cli_main(
            [
                str(image_dir),
                "-o",
                str(raw_mesh_path),
                "--smoothing-iters",
                "0",
                "--no-watertight-check",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert Path(output["raw_mesh_path"]).exists()
    assert Path(output["refined_mesh_path"]).exists()
    assert Path(output["refinement_manifest_path"]).exists()
    assert output["refinement_diagnostics"]["output_point_count"] > 0


@pytest.mark.real_pipeline
def test_real_phase2_phase3_pipeline_with_user_images(tmp_path: Path) -> None:
    env_images = os.environ.get("ZSHIFT_TEST_IMAGES")
    if not env_images:
        pytest.skip("set ZSHIFT_TEST_IMAGES to a folder of user images to run this test")
    image_dir = Path(env_images).resolve()
    if not image_dir.exists():
        pytest.fail(f"ZSHIFT_TEST_IMAGES directory not found: {image_dir}")

    images = collect_input_images(image_dir)
    if len(images) < 2:
        pytest.fail(f"Image directory must contain at least two supported images: {image_dir}")
    output_path = tmp_path / "real_pipeline" / "mesh.obj"
    job = ReconstructionJob(
        mode=ReconstructionMode.MULTI_VIEW,
        label="real_pipeline",
        image_uris=[str(path) for path in images],
        output_path=str(output_path),
        params=Mast3rRunParams(
            device=os.environ.get("ZSHIFT_TEST_DEVICE", "auto"),
            dry_run=False,
        ),
    )

    result = run_phase2_phase3_pipeline(
        job,
        MeshCleaningConfig(smoothing_iters=0, verify_watertight=False),
    )

    assert result.raw_mesh_path.exists()
    assert result.refined_mesh_path.exists()
    assert result.refinement_manifest_path.exists()
    assert result.refinement_diagnostics["output_point_count"] > 0


def test_full_pipeline_viewing_track(fake_reconstruction, tmp_path: Path) -> None:
    import trimesh

    raw_mesh_path = tmp_path / "mesh.obj"
    point_cloud_path = tmp_path / "point_cloud.ply"

    def write_artifacts(raw: Path) -> None:
        pv.Sphere(theta_resolution=16, phi_resolution=16).save(str(raw))
        rng = np.random.default_rng(0)
        points = rng.random((100, 3))
        colors = (rng.random((100, 3)) * 255).astype(np.uint8)
        trimesh.PointCloud(vertices=points, colors=colors).export(str(point_cloud_path))

    fake_reconstruction(raw_mesh_path=raw_mesh_path, write_mesh=write_artifacts)

    result = run_full_pipeline(
        _job(raw_mesh_path),
        use_case="viewing",
        source_type="image_folder",
        refinement_config=MeshCleaningConfig(smoothing_iters=0, verify_watertight=False),
        deliverables_root=tmp_path / "deliverables",
    )

    assert result.pipeline_result.refined_mesh_path.exists()
    assert result.deliverable.track == "viewing"
    assert result.deliverable.output_path is not None
    assert Path(result.deliverable.output_path).exists()
    assert Path(result.deliverable.output_path).suffix == ".ply"


def test_full_pipeline_editing_track(fake_reconstruction, tmp_path: Path) -> None:
    raw_mesh_path = tmp_path / "mesh.obj"
    fake_reconstruction(raw_mesh_path=raw_mesh_path)

    result = run_full_pipeline(
        _job(raw_mesh_path),
        use_case="editing",
        source_type="image_folder",
        refinement_config=MeshCleaningConfig(smoothing_iters=0, verify_watertight=False),
        deliverables_root=tmp_path / "deliverables",
    )

    assert result.pipeline_result.refined_mesh_path.exists()
    assert result.deliverable.track == "editing"
    assert result.deliverable.output_path is not None
    assert Path(result.deliverable.output_path).exists()
    assert Path(result.deliverable.output_path).suffix == ".glb"


def test_full_pipeline_rejects_bad_use_case_for_source_type(monkeypatch, tmp_path: Path) -> None:
    raw_mesh_path = tmp_path / "mesh.obj"

    def fake_reconstruction(job: ReconstructionJob) -> int:
        raise AssertionError("Phase 2 should not run for an invalid route")

    monkeypatch.setattr(
        "spatial_ingestion.final_pipeline.core.run_reconstruction", fake_reconstruction
    )

    from spatial_ingestion.outcomes_engine.engine import InvalidRoutingError

    try:
        run_full_pipeline(
            _job(raw_mesh_path),
            use_case="editing",
            source_type="live_stream",
            refinement_config=MeshCleaningConfig(smoothing_iters=0, verify_watertight=False),
            deliverables_root=tmp_path / "deliverables",
        )
    except InvalidRoutingError:
        pass
    else:
        raise AssertionError("expected invalid source_type/use_case combo to raise")


def test_full_pipeline_rejects_typoed_use_case_before_phase2(monkeypatch, tmp_path: Path) -> None:
    raw_mesh_path = tmp_path / "mesh.obj"

    def fake_reconstruction(job: ReconstructionJob) -> int:
        raise AssertionError("Phase 2 should not run for an unknown use case")

    monkeypatch.setattr(
        "spatial_ingestion.final_pipeline.core.run_reconstruction", fake_reconstruction
    )

    from spatial_ingestion.outcomes_engine.engine import InvalidRoutingError

    try:
        run_full_pipeline(
            _job(raw_mesh_path),
            use_case="liv",
            source_type="image_folder",
            refinement_config=MeshCleaningConfig(smoothing_iters=0, verify_watertight=False),
            deliverables_root=tmp_path / "deliverables",
        )
    except InvalidRoutingError:
        pass
    else:
        raise AssertionError("expected unknown use_case to fail before Phase 2")


def test_full_pipeline_rejects_live_use_case_before_phase2(monkeypatch, tmp_path: Path) -> None:
    raw_mesh_path = tmp_path / "mesh.obj"

    def fake_reconstruction(job: ReconstructionJob) -> int:
        raise AssertionError("Phase 2 should not run for the live use case")

    monkeypatch.setattr(
        "spatial_ingestion.final_pipeline.core.run_reconstruction", fake_reconstruction
    )

    from spatial_ingestion.outcomes_engine.engine import TrackNotImplementedError

    try:
        run_full_pipeline(
            _job(raw_mesh_path),
            use_case="live",
            source_type="live_stream",
            refinement_config=MeshCleaningConfig(smoothing_iters=0, verify_watertight=False),
            deliverables_root=tmp_path / "deliverables",
        )
    except TrackNotImplementedError:
        pass
    else:
        raise AssertionError("expected live use case to fail before Phase 2")


def test_full_pipeline_editing_track_with_glb_artifacts(
    fake_reconstruction, tmp_path: Path
) -> None:
    import trimesh

    raw_mesh_path = tmp_path / "pipeline_out" / "mesh.glb"

    def write_trimesh_sphere(path: Path) -> None:
        sphere = pv.Sphere(theta_resolution=16, phi_resolution=16)
        faces = sphere.faces.reshape(-1, 4)[:, 1:]
        colors = np.zeros((sphere.n_points, 4), dtype=np.uint8)
        colors[:, 0] = 255
        colors[:, 3] = 255
        trimesh.Trimesh(vertices=sphere.points, faces=faces, vertex_colors=colors).export(str(path))

    fake_reconstruction(raw_mesh_path=raw_mesh_path, write_mesh=write_trimesh_sphere)

    result = run_full_pipeline(
        _job(raw_mesh_path),
        use_case="editing",
        source_type="image_folder",
        refinement_config=MeshCleaningConfig(smoothing_iters=0, verify_watertight=False),
        deliverables_root=tmp_path / "deliverables",
    )

    assert result.pipeline_result.raw_mesh_path.suffix == ".glb"
    assert result.pipeline_result.refined_mesh_path.suffix == ".glb"
    assert result.pipeline_result.refined_mesh_path.exists()
    assert result.deliverable.track == "editing"
    assert result.deliverable.output_path is not None
    assert Path(result.deliverable.output_path).exists()
    assert Path(result.deliverable.output_path).suffix == ".glb"
