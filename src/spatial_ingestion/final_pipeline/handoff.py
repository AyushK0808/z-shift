"""Phase 1 -> Phase 2 handoff: batch ingestion schema to reconstruction job."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from spatial_ingestion.auto_rigging.models import AutoRigConfig
from spatial_ingestion.batch_normalization.normalizer import BatchNormalizer
from spatial_ingestion.final_pipeline.core import (
    FinalPipelineResult,
    FullPipelineResult,
    run_full_pipeline,
    run_phase2_phase3_phase5_pipeline,
    run_phase2_phase3_pipeline,
)
from spatial_ingestion.media_classifier.router import (
    MediaClassifierRouter,
    MediaItemDescriptor,
)
from spatial_ingestion.metadata.schema import SourceType, UnifiedSpatialIngestionSchema
from spatial_ingestion.outcomes_engine.engine import DEFAULT_DELIVERABLES_ROOT
from spatial_ingestion.reconstruction.jobs import ReconstructionJobBuilder
from spatial_ingestion.reconstruction.models import Mast3rRunParams, ReconstructionJob
from spatial_ingestion.refinement import MeshCleaningConfig


def ingest_batch(
    paths: Sequence[Path],
    *,
    sync_group_id: str | None = None,
    classifier: MediaClassifierRouter | None = None,
    normalizer: BatchNormalizer | None = None,
) -> UnifiedSpatialIngestionSchema:
    """Run Phase 1 batch ingestion (classify + normalize) in-process.

    Mirrors the gateway's ``POST /v1/ingest/uploads`` flow so the final
    pipeline can consume the same schema the HTTP API produces.
    """
    classifier = classifier or MediaClassifierRouter()
    normalizer = normalizer or BatchNormalizer()
    descriptors = [MediaItemDescriptor(filename=path.name) for path in paths]
    decision = classifier.classify_static(descriptors)
    if decision.input_type == SourceType.UNKNOWN:
        raise ValueError(f"unsupported media: {decision.reason}")
    return normalizer.normalize(list(paths), decision, sync_group_id=sync_group_id)


def load_schema(path: Path | str) -> UnifiedSpatialIngestionSchema:
    """Load a Phase 1 payload from the JSON produced by the ingestion gateway."""
    raw = Path(path).expanduser().read_text(encoding="utf-8")
    return UnifiedSpatialIngestionSchema.model_validate(json.loads(raw))


def build_job(
    payload: UnifiedSpatialIngestionSchema,
    *,
    mast3r_params: Mast3rRunParams | None = None,
    output_path: Path | str | None = None,
    label: str | None = None,
) -> ReconstructionJob:
    """Convert a Phase 1 schema into a Phase 2 reconstruction job."""
    job = ReconstructionJobBuilder().build(payload)
    if mast3r_params:
        params_dump = mast3r_params.model_dump(exclude_unset=True)
        job.metadata = {**job.metadata, **params_dump}
    if output_path:
        job.output_path = str(Path(output_path).expanduser().resolve())
    if label:
        job.label = label
    elif not job.label:
        job.label = payload.sync_group_id or payload.source_type.value
    return job


def run_from_schema(
    payload: UnifiedSpatialIngestionSchema,
    *,
    use_case: str | None = None,
    input_type: str | SourceType | None = None,
    mast3r_params: Mast3rRunParams | None = None,
    output_path: Path | str | None = None,
    refinement_config: MeshCleaningConfig | None = None,
    refined_output_path: Path | str | None = None,
    rig: bool = False,
    rigging_config: AutoRigConfig | None = None,
    rigged_output_path: Path | str | None = None,
    rig_output_dir: Path | str | None = None,
    deliverables_root: Path | str | None = None,
) -> FullPipelineResult | FinalPipelineResult:
    """Run the final pipeline from a Phase 1 payload.

    With ``use_case`` set, runs Phase 2 -> 3 -> 4 (pass ``rig=True`` too, with
    ``use_case="editing"``, to also run Phase 5 and hand Phase 4 the skinned
    GLB instead of the plain refined mesh); otherwise Phase 2 -> 3, or
    Phase 2 -> 3 -> 5 if rigging options are given without a ``use_case``.
    ``input_type`` defaults to the schema's classified source type.
    """
    job = build_job(
        payload,
        mast3r_params=mast3r_params,
        output_path=output_path,
    )
    if use_case:
        return run_full_pipeline(
            job,
            use_case=use_case,
            input_type=input_type or payload.source_type.value,
            refinement_config=refinement_config,
            refined_output_path=refined_output_path,
            deliverables_root=deliverables_root or DEFAULT_DELIVERABLES_ROOT,
            rig=rig,
            rigging_config=rigging_config,
            rigged_output_path=rigged_output_path,
            rig_output_dir=rig_output_dir,
        )
    if rigging_config or rigged_output_path or rig_output_dir:
        return run_phase2_phase3_phase5_pipeline(
            job,
            refinement_config,
            rigging_config=rigging_config,
            refined_output_path=refined_output_path,
            rigged_output_path=rigged_output_path,
            rig_output_dir=rig_output_dir,
        )
    return run_phase2_phase3_pipeline(
        job,
        refinement_config,
        refined_output_path=refined_output_path,
    )


def run_ingested_pipeline(
    paths: Sequence[Path],
    *,
    sync_group_id: str | None = None,
    classifier: MediaClassifierRouter | None = None,
    normalizer: BatchNormalizer | None = None,
    use_case: str | None = None,
    input_type: str | SourceType | None = None,
    mast3r_params: Mast3rRunParams | None = None,
    output_path: Path | str | None = None,
    refinement_config: MeshCleaningConfig | None = None,
    refined_output_path: Path | str | None = None,
    rig: bool = False,
    rigging_config: AutoRigConfig | None = None,
    rigged_output_path: Path | str | None = None,
    rig_output_dir: Path | str | None = None,
    deliverables_root: Path | str | None = None,
) -> FullPipelineResult | FinalPipelineResult:
    """Ingest a batch through Phase 1, then run the final pipeline."""
    payload = ingest_batch(
        paths,
        sync_group_id=sync_group_id,
        classifier=classifier,
        normalizer=normalizer,
    )
    return run_from_schema(
        payload,
        use_case=use_case,
        input_type=input_type,
        mast3r_params=mast3r_params,
        output_path=output_path,
        refinement_config=refinement_config,
        refined_output_path=refined_output_path,
        rig=rig,
        rigging_config=rigging_config,
        rigged_output_path=rigged_output_path,
        rig_output_dir=rig_output_dir,
        deliverables_root=deliverables_root,
    )
