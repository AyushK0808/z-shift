from __future__ import annotations

import json
import os
from pathlib import Path

import pyvista as pv
import pytest

from spatial_ingestion.final_pipeline.cli import main as final_pipeline_cli_main
from spatial_ingestion.final_pipeline.core import PipelineArtifactError, run_phase2_phase3_pipeline
from spatial_ingestion.reconstruction.cli import collect_input_images
from spatial_ingestion.reconstruction.models import ReconstructionJob, ReconstructionMode
from spatial_ingestion.refinement import MeshCleaningConfig


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

    monkeypatch.setattr("spatial_ingestion.final_pipeline.core.run_reconstruction", fake_reconstruction)

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
    (image_dir / "front.png").write_bytes(b"front")
    (image_dir / "side.png").write_bytes(b"side")
    raw_mesh_path = tmp_path / "pipeline_out" / "mesh.obj"

    def fake_reconstruction(job: ReconstructionJob) -> int:
        output_path = Path(job.output_path or raw_mesh_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pv.Sphere(theta_resolution=16, phi_resolution=16).save(str(output_path))
        return 0

    monkeypatch.setattr("spatial_ingestion.final_pipeline.core.run_reconstruction", fake_reconstruction)

    assert final_pipeline_cli_main(
        [
            str(image_dir),
            "-o",
            str(raw_mesh_path),
            "--smoothing-iters",
            "0",
            "--no-watertight-check",
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    assert Path(output["raw_mesh_path"]).exists()
    assert Path(output["refined_mesh_path"]).exists()
    assert Path(output["refinement_manifest_path"]).exists()
    assert output["refinement_diagnostics"]["output_point_count"] > 0


@pytest.mark.real_pipeline
def test_real_phase2_phase3_pipeline_with_user_images(tmp_path: Path) -> None:
    image_dir = Path(
    r"C:\Users\Rakshit\Desktop\OldPCStuff\Mera Saman\CODING\Vinnovate\imgto3d\z-shift\data\pipeline"
).resolve()

    images = collect_input_images(image_dir)
    if len(images) < 2:
        pytest.fail(
        f"Image directory must contain at least two supported images: {image_dir}"
    )
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
