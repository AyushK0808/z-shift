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
    run_phase2_phase3_phase5_pipeline,
    run_phase2_phase3_pipeline,
)
from spatial_ingestion.reconstruction.cli import collect_input_images
from spatial_ingestion.reconstruction.models import ReconstructionJob, ReconstructionMode
from spatial_ingestion.refinement import MeshCleaningConfig
from spatial_ingestion.test_harness.media_factory import create_sample_image


def _job(output_path: Path) -> ReconstructionJob:
    return ReconstructionJob(
        mode=ReconstructionMode.MULTI_VIEW,
        image_uris=["image1.jpg", "image2.jpg"],
        output_path=str(output_path),
    )


def test_pipeline_runs_reconstruction_then_refinement(monkeypatch, tmp_path: Path) -> None:
    raw_mesh_path = tmp_path / "mesh.obj"

    def fake_reconstruction(job: ReconstructionJob) -> int:
        pv.Sphere(theta_resolution=16, phi_resolution=16).save(str(raw_mesh_path))
        return 0

    monkeypatch.setattr(
        "spatial_ingestion.final_pipeline.core.run_reconstruction", fake_reconstruction
    )

    result = run_phase2_phase3_pipeline(
        _job(raw_mesh_path),
        MeshCleaningConfig(smoothing_iters=0, verify_watertight=False),
    )

    assert result.raw_mesh_path == raw_mesh_path.resolve()
    assert result.refined_mesh_path.exists()
    assert result.refinement_manifest_path.exists()
    assert result.refinement_diagnostics["output_point_count"] > 0


def test_pipeline_fails_when_phase2_mesh_is_missing(monkeypatch, tmp_path: Path) -> None:
    raw_mesh_path = tmp_path / "missing.obj"
    monkeypatch.setattr("spatial_ingestion.final_pipeline.core.run_reconstruction", lambda job: 0)

    try:
        run_phase2_phase3_pipeline(_job(raw_mesh_path), MeshCleaningConfig(smoothing_iters=0))
    except PipelineArtifactError as exc:
        assert "no mesh artifact" in str(exc)
    else:
        raise AssertionError("expected missing Phase 2 mesh to fail")


def test_pipeline_cli_smoke_with_mocked_phase2(monkeypatch, tmp_path: Path, capsys) -> None:
    image_dir = tmp_path / "views"
    image_dir.mkdir()
    create_sample_image(image_dir / "front.png")
    create_sample_image(image_dir / "side.png")
    raw_mesh_path = tmp_path / "pipeline_out" / "mesh.obj"

    def fake_reconstruction(job: ReconstructionJob) -> int:
        output_path = Path(job.output_path or raw_mesh_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sphere = pv.Sphere(theta_resolution=16, phi_resolution=16)
        faces = sphere.faces.reshape(-1, 4)[:, 1:]
        import trimesh

        trimesh.Trimesh(vertices=sphere.points, faces=faces).export(str(output_path))
        return 0

    monkeypatch.setattr(
        "spatial_ingestion.final_pipeline.core.run_reconstruction", fake_reconstruction
    )

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


def test_pipeline_runs_phase5_after_refinement(monkeypatch, tmp_path: Path) -> None:
    raw_mesh_path = tmp_path / "pipeline_out" / "mesh.glb"
    rigged_output = tmp_path / "paper" / "rigged.glb"

    def fake_reconstruction(job: ReconstructionJob) -> int:
        output_path = Path(job.output_path or raw_mesh_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sphere = pv.Sphere(theta_resolution=16, phi_resolution=16)
        faces = sphere.faces.reshape(-1, 4)[:, 1:]
        colors = np.zeros((sphere.n_points, 4), dtype=np.uint8)
        colors[:, 1] = 255
        colors[:, 3] = 255
        import trimesh

        trimesh.Trimesh(vertices=sphere.points, faces=faces, vertex_colors=colors).export(
            str(output_path)
        )
        return 0

    monkeypatch.setattr(
        "spatial_ingestion.final_pipeline.core.run_reconstruction", fake_reconstruction
    )

    result = run_phase2_phase3_phase5_pipeline(
        _job(raw_mesh_path),
        MeshCleaningConfig(smoothing_iters=0, verify_watertight=False),
        rigged_output_path=rigged_output,
        rig_output_dir=tmp_path / "paper",
    )

    assert result.refined_mesh_path.exists()
    assert result.rigged_mesh_path == rigged_output
    assert result.rigged_mesh_path.exists()
    assert result.skeleton_path is not None and result.skeleton_path.exists()
    assert result.skinning_weights_path is not None and result.skinning_weights_path.exists()
    assert result.rigging_manifest_path is not None and result.rigging_manifest_path.exists()


def test_pipeline_cli_rig_outputs_phase5_artifacts(monkeypatch, tmp_path: Path, capsys) -> None:
    image_dir = tmp_path / "views"
    image_dir.mkdir()
    create_sample_image(image_dir / "front.png")
    create_sample_image(image_dir / "side.png")
    raw_mesh_path = tmp_path / "pipeline_out" / "mesh.glb"
    rigged_output = tmp_path / "paper" / "rigged.glb"

    def fake_reconstruction(job: ReconstructionJob) -> int:
        output_path = Path(job.output_path or raw_mesh_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sphere = pv.Sphere(theta_resolution=16, phi_resolution=16)
        faces = sphere.faces.reshape(-1, 4)[:, 1:]
        import trimesh

        trimesh.Trimesh(vertices=sphere.points, faces=faces).export(str(output_path))
        return 0

    monkeypatch.setattr(
        "spatial_ingestion.final_pipeline.core.run_reconstruction", fake_reconstruction
    )

    assert (
        final_pipeline_cli_main(
            [
                str(image_dir),
                "-o",
                str(raw_mesh_path),
                "--smoothing-iters",
                "0",
                "--no-watertight-check",
                "--rig",
                "--articulation",
                "biped",
                "--rig-output-dir",
                str(tmp_path / "paper"),
                "--rigged-output",
                str(rigged_output),
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert Path(output["raw_mesh_path"]).exists()
    assert Path(output["refined_mesh_path"]).exists()
    assert Path(output["rigged_mesh_path"]) == rigged_output
    assert Path(output["rigged_mesh_path"]).exists()
    assert Path(output["rigging_manifest_path"]).exists()


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
        metadata={
            "device": os.environ.get("ZSHIFT_TEST_DEVICE", "auto"),
            "dry_run": False,
        },
    )

    result = run_phase2_phase3_pipeline(
        job,
        MeshCleaningConfig(smoothing_iters=0, verify_watertight=False),
    )

    assert result.raw_mesh_path.exists()
    assert result.refined_mesh_path.exists()
    assert result.refinement_manifest_path.exists()
    assert result.refinement_diagnostics["output_point_count"] > 0


def test_full_pipeline_viewing_track(monkeypatch, tmp_path: Path) -> None:
    import trimesh

    raw_mesh_path = tmp_path / "mesh.obj"
    point_cloud_path = tmp_path / "point_cloud.ply"

    def fake_reconstruction(job: ReconstructionJob) -> int:
        pv.Sphere(theta_resolution=16, phi_resolution=16).save(str(raw_mesh_path))
        rng = np.random.default_rng(0)
        points = rng.random((100, 3))
        colors = (rng.random((100, 3)) * 255).astype(np.uint8)
        trimesh.PointCloud(vertices=points, colors=colors).export(str(point_cloud_path))
        return 0

    monkeypatch.setattr(
        "spatial_ingestion.final_pipeline.core.run_reconstruction", fake_reconstruction
    )

    result = run_full_pipeline(
        _job(raw_mesh_path),
        use_case="viewing",
        input_type="image_folder",
        refinement_config=MeshCleaningConfig(smoothing_iters=0, verify_watertight=False),
        deliverables_root=tmp_path / "deliverables",
    )

    assert result.pipeline_result.refined_mesh_path.exists()
    assert result.deliverable.track == "B"
    assert result.deliverable.output_path is not None
    assert Path(result.deliverable.output_path).exists()
    assert Path(result.deliverable.output_path).suffix == ".ply"


def test_full_pipeline_editing_track(monkeypatch, tmp_path: Path) -> None:
    raw_mesh_path = tmp_path / "mesh.obj"

    def fake_reconstruction(job: ReconstructionJob) -> int:
        pv.Sphere(theta_resolution=16, phi_resolution=16).save(str(raw_mesh_path))
        return 0

    monkeypatch.setattr(
        "spatial_ingestion.final_pipeline.core.run_reconstruction", fake_reconstruction
    )

    result = run_full_pipeline(
        _job(raw_mesh_path),
        use_case="editing",
        input_type="image_folder",
        refinement_config=MeshCleaningConfig(smoothing_iters=0, verify_watertight=False),
        deliverables_root=tmp_path / "deliverables",
    )

    assert result.pipeline_result.refined_mesh_path.exists()
    assert result.deliverable.track == "A"
    assert result.deliverable.output_path is not None
    assert Path(result.deliverable.output_path).exists()
    assert Path(result.deliverable.output_path).suffix == ".glb"


def test_full_pipeline_rejects_bad_use_case_for_input_type(monkeypatch, tmp_path: Path) -> None:
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
            input_type="live_stream",
            refinement_config=MeshCleaningConfig(smoothing_iters=0, verify_watertight=False),
            deliverables_root=tmp_path / "deliverables",
        )
    except InvalidRoutingError:
        pass
    else:
        raise AssertionError("expected invalid input_type/use_case combo to raise")


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
            input_type="image_folder",
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
            input_type="live_stream",
            refinement_config=MeshCleaningConfig(smoothing_iters=0, verify_watertight=False),
            deliverables_root=tmp_path / "deliverables",
        )
    except TrackNotImplementedError:
        pass
    else:
        raise AssertionError("expected live use case to fail before Phase 2")


def test_full_pipeline_editing_track_with_glb_artifacts(monkeypatch, tmp_path: Path) -> None:
    import trimesh

    raw_mesh_path = tmp_path / "pipeline_out" / "mesh.glb"

    def fake_reconstruction(job: ReconstructionJob) -> int:
        output_path = Path(job.output_path or raw_mesh_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sphere = pv.Sphere(theta_resolution=16, phi_resolution=16)
        faces = sphere.faces.reshape(-1, 4)[:, 1:]
        colors = np.zeros((sphere.n_points, 4), dtype=np.uint8)
        colors[:, 0] = 255
        colors[:, 3] = 255
        trimesh.Trimesh(vertices=sphere.points, faces=faces, vertex_colors=colors).export(
            str(output_path)
        )
        return 0

    monkeypatch.setattr(
        "spatial_ingestion.final_pipeline.core.run_reconstruction", fake_reconstruction
    )

    result = run_full_pipeline(
        _job(raw_mesh_path),
        use_case="editing",
        input_type="image_folder",
        refinement_config=MeshCleaningConfig(smoothing_iters=0, verify_watertight=False),
        deliverables_root=tmp_path / "deliverables",
    )

    assert result.pipeline_result.raw_mesh_path.suffix == ".glb"
    assert result.pipeline_result.refined_mesh_path.suffix == ".glb"
    assert result.pipeline_result.refined_mesh_path.exists()
    assert result.deliverable.track == "A"
    assert result.deliverable.output_path is not None
    assert Path(result.deliverable.output_path).exists()
    assert Path(result.deliverable.output_path).suffix == ".glb"


def test_full_pipeline_editing_track_with_rig_delivers_skinned_glb(
    monkeypatch, tmp_path: Path
) -> None:
    import trimesh

    from spatial_ingestion.auto_rigging.models import ArticulationType, AutoRigConfig

    raw_mesh_path = tmp_path / "pipeline_out" / "mesh.glb"

    def fake_reconstruction(job: ReconstructionJob) -> int:
        output_path = Path(job.output_path or raw_mesh_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sphere = pv.Sphere(theta_resolution=16, phi_resolution=16)
        faces = sphere.faces.reshape(-1, 4)[:, 1:]
        trimesh.Trimesh(vertices=sphere.points, faces=faces).export(str(output_path))
        return 0

    monkeypatch.setattr(
        "spatial_ingestion.final_pipeline.core.run_reconstruction", fake_reconstruction
    )

    result = run_full_pipeline(
        _job(raw_mesh_path),
        use_case="editing",
        input_type="image_folder",
        refinement_config=MeshCleaningConfig(smoothing_iters=0, verify_watertight=False),
        deliverables_root=tmp_path / "deliverables",
        rig=True,
        rigging_config=AutoRigConfig(articulation_type=ArticulationType.BIPED),
    )

    assert result.pipeline_result.rigged_mesh_path is not None
    assert result.pipeline_result.rigged_mesh_path.exists()
    assert result.deliverable.track == "A"
    assert result.deliverable.output_path is not None
    delivered_path = Path(result.deliverable.output_path)
    assert delivered_path.exists()
    assert delivered_path.suffix == ".glb"
    assert delivered_path.name.endswith("_rigged.glb")
    # The delivered file must be the actual rigged GLB (copied byte-for-byte),
    # not a trimesh re-export that would silently drop the skin data.
    assert delivered_path.read_bytes() == result.pipeline_result.rigged_mesh_path.read_bytes()


def test_full_pipeline_rejects_rig_with_viewing_use_case(monkeypatch, tmp_path: Path) -> None:
    raw_mesh_path = tmp_path / "mesh.obj"

    def fake_reconstruction(job: ReconstructionJob) -> int:
        raise AssertionError("Phase 2 should not run for an invalid --rig/use_case combo")

    monkeypatch.setattr(
        "spatial_ingestion.final_pipeline.core.run_reconstruction", fake_reconstruction
    )

    from spatial_ingestion.outcomes_engine.engine import InvalidRoutingError

    try:
        run_full_pipeline(
            _job(raw_mesh_path),
            use_case="viewing",
            input_type="image_folder",
            refinement_config=MeshCleaningConfig(smoothing_iters=0, verify_watertight=False),
            deliverables_root=tmp_path / "deliverables",
            rig=True,
        )
    except InvalidRoutingError:
        pass
    else:
        raise AssertionError("expected --rig combined with use_case='viewing' to fail")


def test_pipeline_cli_rejects_rig_with_viewing_use_case(tmp_path: Path) -> None:
    image_dir = tmp_path / "views"
    image_dir.mkdir()
    create_sample_image(image_dir / "front.png")
    create_sample_image(image_dir / "side.png")

    with pytest.raises(SystemExit):
        final_pipeline_cli_main(
            [
                str(image_dir),
                "-o",
                str(tmp_path / "pipeline_out" / "mesh.glb"),
                "--use-case",
                "viewing",
                "--rig",
            ]
        )
