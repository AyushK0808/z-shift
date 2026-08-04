from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import pyvista as pv

from spatial_ingestion.batch_normalization.image_processor import ImageProcessor
from spatial_ingestion.batch_normalization.normalizer import BatchNormalizer
from spatial_ingestion.final_pipeline.cli import main as final_pipeline_cli_main
from spatial_ingestion.final_pipeline.core import FinalPipelineResult, FullPipelineResult
from spatial_ingestion.final_pipeline.handoff import (
    build_job,
    ingest_batch,
    load_schema,
    run_from_schema,
    run_ingested_pipeline,
)
from spatial_ingestion.metadata.schema import (
    FrameReference,
    SourceType,
    Track,
    UnifiedSpatialIngestionSchema,
)
from spatial_ingestion.reconstruction._io import uri_to_path
from spatial_ingestion.reconstruction.models import (
    Mast3rRunParams,
    ReconstructionJob,
    ReconstructionMode,
)
from spatial_ingestion.refinement import MeshCleaningConfig
from spatial_ingestion.test_harness.media_factory import create_sample_image


def _normalizer(output_root: Path) -> BatchNormalizer:
    return BatchNormalizer(image_processor=ImageProcessor(output_root=output_root))


def _ingest_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "views"
    folder.mkdir()
    create_sample_image(folder / "front.jpg")
    create_sample_image(folder / "side.jpg")
    return folder


def _fake_reconstruction(monkeypatch, raw_mesh_path: Path):
    def fake_reconstruction(job: ReconstructionJob) -> int:
        image_paths = [uri_to_path(uri) for uri in job.image_uris]
        assert all(path.exists() for path in image_paths)
        raw = Path(job.output_path or raw_mesh_path).resolve()
        raw.parent.mkdir(parents=True, exist_ok=True)
        pv.Sphere(theta_resolution=16, phi_resolution=16).save(str(raw))
        return 0

    monkeypatch.setattr(
        "spatial_ingestion.final_pipeline.core.run_reconstruction", fake_reconstruction
    )
    return fake_reconstruction


def test_ingest_batch_produces_phase1_schema(tmp_path: Path) -> None:
    folder = _ingest_folder(tmp_path)
    payload = ingest_batch(
        sorted(folder.iterdir()), normalizer=_normalizer(tmp_path / "normalized")
    )

    assert payload.source_type == SourceType.IMAGE_FOLDER
    assert payload.frame_count == 2
    assert len(payload.frames) == 2
    for frame in payload.frames:
        assert frame.uri is not None
        assert frame.uri.startswith("file://")
        assert frame.resolution == (320, 240)
        assert uri_to_path(frame.uri).exists()


def test_build_job_merges_mast3r_params(tmp_path: Path) -> None:
    folder = _ingest_folder(tmp_path)
    payload = ingest_batch(
        sorted(folder.iterdir()), normalizer=_normalizer(tmp_path / "normalized")
    )

    job = build_job(
        payload,
        mast3r_params=Mast3rRunParams(model_name="test/model", image_size=224),
        output_path=tmp_path / "out" / "mesh.obj",
        label="custom_label",
    )

    assert job.mode == ReconstructionMode.MULTI_VIEW
    assert job.label == "custom_label"
    # build_job resolves the output path with the job id so folder and job match.
    assert job.output_path is not None
    assert job.output_path.endswith("mesh.obj")
    assert f"mesh_{job.job_id}" in job.output_path
    assert job.metadata["model_name"] == "test/model"
    assert job.metadata["image_size"] == 224
    assert job.metadata["source_type"] == "image_folder"
    assert len(job.image_uris) == 2


def test_run_from_schema_pipeline_with_file_uris(monkeypatch, tmp_path: Path) -> None:
    folder = _ingest_folder(tmp_path)
    payload = ingest_batch(
        sorted(folder.iterdir()), normalizer=_normalizer(tmp_path / "normalized")
    )
    raw_mesh_path = tmp_path / "mesh.obj"
    _fake_reconstruction(monkeypatch, raw_mesh_path)

    result = run_from_schema(
        payload,
        mast3r_params=Mast3rRunParams(device="cpu"),
        output_path=tmp_path / "out" / "mesh.obj",
        refinement_config=MeshCleaningConfig(smoothing_iters=0, verify_watertight=False),
    )

    assert isinstance(result, FinalPipelineResult)
    assert result.job_id
    assert result.raw_mesh_path.exists()
    assert result.refined_mesh_path.exists()
    assert result.refinement_manifest_path.exists()
    assert result.refinement_diagnostics["output_point_count"] > 0


def test_run_from_schema_with_use_case_defaults_input_type(monkeypatch, tmp_path: Path) -> None:
    folder = _ingest_folder(tmp_path)
    payload = ingest_batch(
        sorted(folder.iterdir()), normalizer=_normalizer(tmp_path / "normalized")
    )
    _fake_reconstruction(monkeypatch, tmp_path / "mesh.obj")

    result = run_from_schema(
        payload,
        use_case="editing",
        output_path=tmp_path / "out" / "mesh.obj",
        refinement_config=MeshCleaningConfig(smoothing_iters=0, verify_watertight=False),
        deliverables_root=tmp_path / "deliverables",
    )

    assert isinstance(result, FullPipelineResult)
    assert result.deliverable.track == "A"
    assert result.deliverable.output_path is not None
    assert Path(result.deliverable.output_path).exists()


def test_run_ingested_pipeline(monkeypatch, tmp_path: Path) -> None:
    folder = _ingest_folder(tmp_path)
    _fake_reconstruction(monkeypatch, tmp_path / "mesh.obj")

    result = run_ingested_pipeline(
        sorted(folder.iterdir()),
        normalizer=_normalizer(tmp_path / "normalized"),
        output_path=tmp_path / "out" / "mesh.obj",
        refinement_config=MeshCleaningConfig(smoothing_iters=0, verify_watertight=False),
    )

    assert isinstance(result, FinalPipelineResult)
    assert result.job_id
    assert result.refined_mesh_path.exists()


def test_pipeline_cli_from_schema(monkeypatch, tmp_path: Path, capsys) -> None:
    folder = _ingest_folder(tmp_path)
    payload = ingest_batch(
        sorted(folder.iterdir()), normalizer=_normalizer(tmp_path / "normalized")
    )
    schema_path = tmp_path / "payload.json"
    schema_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    raw_mesh_path = tmp_path / "out" / "mesh.obj"
    _fake_reconstruction(monkeypatch, raw_mesh_path)

    assert (
        final_pipeline_cli_main(
            [
                str(folder),
                "--from-schema",
                str(schema_path),
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


def test_load_schema_round_trips_gateway_json(tmp_path: Path) -> None:
    folder = _ingest_folder(tmp_path)
    payload = ingest_batch(
        sorted(folder.iterdir()), normalizer=_normalizer(tmp_path / "normalized")
    )
    schema_path = tmp_path / "payload.json"
    schema_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")

    loaded = load_schema(schema_path)

    assert loaded.source_type == payload.source_type
    assert loaded.frame_count == payload.frame_count
    assert loaded.frames[0].uri == payload.frames[0].uri


def test_build_job_keeps_builder_pairing_recommendation() -> None:
    payload = _single_video_payload()

    job = build_job(payload, mast3r_params=Mast3rRunParams())

    assert job.mode == ReconstructionMode.VIDEO_SEQUENCE
    assert job.metadata["pairing_strategy"] == "swin"
    resolved = Mast3rRunParams.model_validate(job.metadata)
    assert resolved.model_name == Mast3rRunParams().model_name


def test_build_job_explicit_pairing_overrides_builder_recommendation() -> None:
    payload = _single_video_payload()

    job = build_job(payload, mast3r_params=Mast3rRunParams(pairing_strategy="complete"))

    assert job.metadata["pairing_strategy"] == "complete"


def _single_video_payload() -> UnifiedSpatialIngestionSchema:
    return UnifiedSpatialIngestionSchema(
        source_type=SourceType.SINGLE_VIDEO,
        track=Track.BATCH,
        resolution=(1024, 1024),
        frame_count=3,
        is_stream=False,
        compute_priority_score=0.5,
        frames=[
            FrameReference(
                frame_id=f"frame_{index}",
                uri=f"file:///tmp/frame_{index}.jpg",
                index=index,
                source_id="cam_a",
            )
            for index in range(3)
        ],
    )


def test_uri_to_path_handles_drive_netloc_form() -> None:
    if sys.platform != "win32":
        pytest.skip("Windows drive-letter URI form only")
    assert str(uri_to_path("file://d:/tmp/a.png")).replace("\\", "/") == "d:/tmp/a.png"
