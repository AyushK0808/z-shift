"""Integration tests for the Phase 1 -> 2 -> 3 -> 4 handoffs.

These cover the boundaries the unit tests mock away:
- the ingestion gateway persists its payload for downstream consumption,
- the reconstruction job id is threaded into Phase 4 deliverables,
- the Phase 2 export path really produces the mesh + point cloud the later
  phases consume (with MASt3R alignment stubbed, since the alignment model
  download is not part of the test suite),
- clear failures for single-image, live, and unsupported-format inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pytest
import trimesh

from spatial_ingestion.batch_normalization.image_processor import ImageProcessor
from spatial_ingestion.batch_normalization.normalizer import BatchNormalizer
from spatial_ingestion.final_pipeline.cli import main as final_pipeline_cli_main
from spatial_ingestion.final_pipeline.core import run_full_pipeline
from spatial_ingestion.final_pipeline.handoff import build_job, ingest_batch, load_schema
from spatial_ingestion.ingestion_gateway.api import create_app
from spatial_ingestion.media_classifier.router import (
    MediaClassifierRouter,
    MediaItemDescriptor,
)
from spatial_ingestion.metadata.schema import (
    SourceType,
    Track,
    UnifiedSpatialIngestionSchema,
)
from spatial_ingestion.outcomes_engine.engine import deliverable_router
from spatial_ingestion.reconstruction.cli import resolve_output_path
from spatial_ingestion.reconstruction.models import (
    Mast3rRunParams,
    ReconstructionJob,
    ReconstructionMode,
)
from spatial_ingestion.reconstruction.pipeline import run as run_reconstruction
from spatial_ingestion.refinement import MeshCleaningConfig
from spatial_ingestion.storage.object_store import ObjectStore
from spatial_ingestion.storage.payload_store import PayloadStore
from spatial_ingestion.test_harness.media_factory import create_sample_image

# ---------------------------------------------------------------------------
# Phase 1 -> 2 handoff: gateway persists the payload
# ---------------------------------------------------------------------------


def _ingest_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "views"
    folder.mkdir()
    create_sample_image(folder / "front.jpg")
    create_sample_image(folder / "side.jpg")
    return folder


def _normalizer(output_root: Path) -> BatchNormalizer:
    return BatchNormalizer(image_processor=ImageProcessor(output_root=output_root))


def _require_mast3r_stack() -> None:
    """Skip the real-export tests unless the heavy MASt3R stack is importable.

    The export path imports `mast3r.cloud_opt.tsdf_optimizer` and
    `dust3r.utils.device` unconditionally; without them it raises RuntimeError
    rather than degrade gracefully, so gate the tests on their presence.
    """
    pytest.importorskip("mast3r")
    pytest.importorskip("dust3r")


def test_gateway_upload_persists_payload_for_downstream(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    folder = _ingest_folder(tmp_path)
    app = create_app()
    app.state.gateway_state.batch_normalizer = _normalizer(tmp_path / "normalized")
    app.state.gateway_state.object_store = ObjectStore(root=tmp_path / "object_store")
    app.state.gateway_state.payload_store = PayloadStore(root=tmp_path / "payloads")

    client = TestClient(app)
    files = []
    try:
        with (folder / "front.jpg").open("rb") as front, (folder / "side.jpg").open("rb") as side:
            files = [
                ("files", ("front.jpg", front, "image/jpeg")),
                ("files", ("side.jpg", side, "image/jpeg")),
            ]
            response = client.post(
                "/v1/ingest/uploads", files=files, headers={"authorization": "Bearer dev-token"}
            )
    finally:
        for _, (_, handle, _) in files:
            handle.close()

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["source_type"] == SourceType.IMAGE_FOLDER.value
    assert payload["frame_count"] == 2

    payload_uri = payload["metadata"]["payload_uri"]
    assert payload_uri.startswith("file://")
    stored = Path(urlparse(payload_uri).path)
    assert stored.exists()

    loaded = load_schema(stored)
    assert loaded.source_type == payload["source_type"]
    assert loaded.frame_count == payload["frame_count"]
    assert loaded.frames[0].uri == payload["frames"][0]["uri"]
    # The persisted file is self-describing: its own metadata carries the uri.
    assert loaded.metadata["payload_uri"] == payload_uri


# ---------------------------------------------------------------------------
# Phase 2 -> 3 -> 4: real pipeline.run chain with alignment stubbed
# ---------------------------------------------------------------------------


class _FakeScene:
    """Minimal MASt3R scene returning a synthetic dense depth field."""

    def __init__(self) -> None:
        h, w = 32, 32
        yy, xx = np.mgrid[0:h, 0:w]
        z = (np.sin(xx / 8.0) * np.cos(yy / 8.0) * 0.5).astype(np.float32)
        self.imgs = [np.full((h, w, 3), 0.7, dtype=np.float32) for _ in range(2)]
        self._pts3d = [
            np.stack([xx.astype(np.float32), yy.astype(np.float32), z * (i + 1)], axis=-1)
            for i in range(2)
        ]
        self._confs = [np.full((h, w), 2.0, dtype=np.float32) for _ in range(2)]

    def get_dense_pts3d(self, clean_depth: bool = True):
        return self._pts3d, None, self._confs


def _job_from_folder(tmp_path: Path) -> ReconstructionJob:
    folder = _ingest_folder(tmp_path)
    payload = ingest_batch(
        sorted(folder.iterdir()), normalizer=_normalizer(tmp_path / "normalized")
    )
    job = build_job(payload, mast3r_params=Mast3rRunParams(device="cpu", image_size=512))
    job.output_path = str(tmp_path / "out" / "mesh.glb")
    return job


@pytest.mark.parametrize("use_case", ["editing", "viewing"])
def test_real_pipeline_chain_round_trips_job_id(monkeypatch, tmp_path: Path, use_case: str) -> None:
    _require_mast3r_stack()

    def fake_alignment(**kwargs):
        return _FakeScene()

    monkeypatch.setattr(
        "spatial_ingestion.reconstruction.pipeline.run_sparse_alignment", fake_alignment
    )

    job = _job_from_folder(tmp_path)
    assert job.output_path is not None

    # Phase 2 runs for real (export + manifest + point_cloud.ply), with only
    # the MASt3R alignment call stubbed.
    run_result = run_reconstruction(job)
    assert run_result.output_path == Path(job.output_path).resolve()
    assert run_result.output_path.exists()
    assert run_result.point_cloud_path.exists()
    manifest = json.loads(run_result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["job_id"] == job.job_id

    result = run_full_pipeline(
        job,
        use_case=use_case,
        source_type="image_folder",
        refinement_config=MeshCleaningConfig(smoothing_iters=0, verify_watertight=False),
        deliverables_root=tmp_path / "deliverables",
    )

    assert result.pipeline_result.job_id == job.job_id
    assert result.deliverable.job_id == job.job_id, (
        "Phase 4 deliverable must reuse the Phase 2 job id so manifests, "
        "folders, and deliverables are traceable to the same run"
    )
    assert result.deliverable.output_path is not None
    assert Path(result.deliverable.output_path).exists()

    # Phase 2 really produced geometry (not the old random icosphere mock).
    assert result.pipeline_result.raw_mesh_path.exists()
    assert result.pipeline_result.refined_mesh_path.exists()


def test_real_pipeline_viewing_uses_phase2_point_cloud(monkeypatch, tmp_path: Path) -> None:
    _require_mast3r_stack()

    def fake_alignment(**kwargs):
        return _FakeScene()

    monkeypatch.setattr(
        "spatial_ingestion.reconstruction.pipeline.run_sparse_alignment", fake_alignment
    )

    job = _job_from_folder(tmp_path)
    assert job.output_path is not None

    run_result = run_reconstruction(job)
    point_cloud = trimesh.load(str(run_result.point_cloud_path))
    assert isinstance(point_cloud, trimesh.PointCloud)
    assert len(point_cloud.vertices) > 100


# ---------------------------------------------------------------------------
# Phase 4 contract: no fabricated geometry, id threading
# ---------------------------------------------------------------------------


def test_deliverable_router_requires_real_artifacts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires a Phase 3 cleaned mesh"):
        deliverable_router("image_folder", "editing", output_root=tmp_path)
    with pytest.raises(ValueError, match="requires a Phase 3 point-cloud"):
        deliverable_router("image_folder", "viewing", output_root=tmp_path)


def test_deliverable_router_respects_explicit_job_id(tmp_path: Path) -> None:
    mesh = trimesh.creation.icosphere(subdivisions=1)
    result = deliverable_router(
        "image_folder",
        "editing",
        output_root=tmp_path,
        job_id="traceable_job",
        mesh=mesh,
    )
    assert result.job_id == "traceable_job"
    assert Path(result.output_path or "").name == "traceable_job_model.glb"


# ---------------------------------------------------------------------------
# CLI boundary failures are clear and early
# ---------------------------------------------------------------------------


def test_cli_live_use_case_fails_before_ingestion(tmp_path: Path, capsys) -> None:
    folder = _ingest_folder(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        final_pipeline_cli_main([str(folder), "--use-case", "live"])
    assert excinfo.value.code == 2
    assert "live delivery" in capsys.readouterr().err


def test_cli_from_schema_rejects_single_image(tmp_path: Path, capsys) -> None:
    folder = tmp_path / "single"
    folder.mkdir()
    image_path = folder / "only.jpg"
    create_sample_image(image_path)
    decision = MediaClassifierRouter().classify_static(
        [MediaItemDescriptor(filename=image_path.name)]
    )
    payload = BatchNormalizer(
        image_processor=ImageProcessor(output_root=tmp_path / "normalized")
    ).normalize([image_path], decision)
    schema_path = tmp_path / "single.json"
    schema_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        final_pipeline_cli_main(["--from-schema", str(schema_path)])
    assert excinfo.value.code == 2
    assert "single-image reconstruction is not supported" in capsys.readouterr().err


def test_build_job_rejects_live_stream_payload() -> None:
    payload = UnifiedSpatialIngestionSchema(
        source_type=SourceType.LIVE_STREAM,
        track=Track.LIVE,
        is_stream=True,
        compute_priority_score=1.0,
    )
    with pytest.raises(ValueError, match="live streams cannot be converted"):
        build_job(payload)


def test_resolve_output_path_rejects_unsupported_phase2_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported Phase 2 mesh format"):
        resolve_output_path(tmp_path, str(tmp_path / "mesh.stl"), job_id="abc")


def test_build_job_embeds_job_id_in_programmatic_output_folder(tmp_path: Path) -> None:
    """run_from_schema/run_ingested_pipeline (via build_job) must resolve output
    the same way the CLI does — folder name carries the job id."""
    folder = _ingest_folder(tmp_path)
    payload = ingest_batch(
        sorted(folder.iterdir()), normalizer=_normalizer(tmp_path / "normalized")
    )
    job = build_job(payload, output_path=tmp_path / "out" / "mesh.obj", label="prog")
    assert job.output_path is not None
    assert job.output_path.endswith("mesh.obj")
    assert f"mesh_{job.job_id}" in job.output_path


def test_build_job_rejects_unsupported_format_via_programmatic_path(tmp_path: Path) -> None:
    folder = _ingest_folder(tmp_path)
    payload = ingest_batch(
        sorted(folder.iterdir()), normalizer=_normalizer(tmp_path / "normalized")
    )
    with pytest.raises(ValueError, match="Unsupported Phase 2 mesh format"):
        build_job(payload, output_path=tmp_path / "out" / "mesh.stl")


def test_pipeline_run_rejects_unsupported_format_for_hand_built_job(tmp_path: Path) -> None:
    job = ReconstructionJob(
        mode=ReconstructionMode.MULTI_VIEW,
        image_uris=["a.jpg", "b.jpg"],
        output_path=str(tmp_path / "out" / "mesh.stl"),
    )
    with pytest.raises(ValueError, match="Unsupported Phase 2 mesh format"):
        run_reconstruction(job)


def test_cli_folder_with_single_image_is_clean_exit(tmp_path: Path, capsys) -> None:
    folder = tmp_path / "views"
    folder.mkdir()
    create_sample_image(folder / "only.jpg")
    with pytest.raises(SystemExit) as excinfo:
        final_pipeline_cli_main([str(folder)])
    assert excinfo.value.code == 2
    assert "at least two views" in capsys.readouterr().err
