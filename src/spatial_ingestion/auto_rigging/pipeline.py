from __future__ import annotations

import logging
from pathlib import Path

import trimesh

from spatial_ingestion.auto_rigging.export import RigMetadataExporter
from spatial_ingestion.auto_rigging.models import AutoRigConfig, AutoRigResult, RiggedMesh
from spatial_ingestion.auto_rigging.skeleton.templates import TemplateSkeletonFitter
from spatial_ingestion.auto_rigging.skinning.inverse_distance import InverseDistanceSkinner

logger = logging.getLogger(__name__)


class AutoRiggingPipeline:
    """Phase 5 MVP: mesh -> template skeleton -> inverse-distance skinning."""

    def __init__(
        self,
        skeleton_fitter: TemplateSkeletonFitter | None = None,
        skinner: InverseDistanceSkinner | None = None,
        exporter: RigMetadataExporter | None = None,
    ) -> None:
        self._skeleton_fitter = skeleton_fitter or TemplateSkeletonFitter()
        self._skinner = skinner or InverseDistanceSkinner()
        self._exporter = exporter or RigMetadataExporter()

    def rig_mesh(
        self,
        mesh: trimesh.Trimesh,
        config: AutoRigConfig | None = None,
        mesh_uri: str | None = None,
        export_metadata: bool = True,
    ) -> AutoRigResult:
        cfg = config or AutoRigConfig()
        logger.info(
            "Rigging mesh: vertices=%d, faces=%d, articulation_type=%s",
            len(mesh.vertices),
            len(mesh.faces),
            cfg.articulation_type.value,
        )
        working_mesh = self._prepare_mesh(mesh, cfg)
        skeleton = self._skeleton_fitter.fit(working_mesh, cfg.articulation_type)
        logger.info(
            "Fitted skeleton: joints=%d, bones=%d, root=%s",
            len(skeleton.joints),
            len(skeleton.bones),
            skeleton.root_joint,
        )
        skinning = self._skinner.compute(
            working_mesh,
            skeleton,
            max_influences=cfg.max_skinning_influences,
        )
        logger.info(
            "Computed skinning weights: vertices=%d, max_influences=%d",
            len(skinning.weights),
            skinning.max_influences,
        )
        rigged_mesh = RiggedMesh(
            mesh_uri=mesh_uri,
            vertex_count=int(len(working_mesh.vertices)),
            face_count=int(len(working_mesh.faces)),
            skeleton=skeleton,
            skinning=skinning,
            metadata={
                "articulation_type": cfg.articulation_type.value,
                "skinning_method": "inverse_distance_to_joints",
                "skeleton_method": "template_bbox_fit",
                "template_axis_convention": "Y-up height, X forward/width, Z lateral/depth",
            },
        )
        result = AutoRigResult(
            rigged_mesh=rigged_mesh,
            warnings=[
                "Template skeleton fitting assumes a world-axis-aligned, Y-up mesh; "
                "joints may be misplaced for arbitrary orientations."
            ],
        )
        for warning in result.warnings:
            logger.warning(warning)
        if export_metadata:
            exporter = (
                RigMetadataExporter(cfg.output_dir)
                if cfg.output_dir is not None
                else self._exporter
            )
            bundle = exporter.export_bundle(
                working_mesh,
                result,
                rigged_mesh_path=cfg.rigged_output_path,
            )
            logger.info(
                "Rigging export complete: rigged_mesh=%s, manifest=%s",
                bundle.rigged_mesh_uri,
                bundle.manifest_uri,
            )
            return bundle
        return result

    def rig_mesh_file(
        self,
        mesh_path: Path,
        config: AutoRigConfig | None = None,
        export_metadata: bool = True,
    ) -> AutoRigResult:
        loaded = trimesh.load_mesh(mesh_path, process=False)
        if not isinstance(loaded, trimesh.Trimesh):
            raise ValueError(f"Expected a single mesh at {mesh_path}")
        return self.rig_mesh(
            loaded,
            config=config,
            mesh_uri=mesh_path.resolve().as_uri(),
            export_metadata=export_metadata,
        )

    @staticmethod
    def _prepare_mesh(mesh: trimesh.Trimesh, config: AutoRigConfig) -> trimesh.Trimesh:
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError("mesh must contain vertices and faces")
        if not config.normalize_mesh:
            return mesh.copy()

        normalized = mesh.copy()
        extents = normalized.extents
        scale = float(max(extents)) if len(extents) else 1.0
        if scale <= 0:
            return normalized
        normalized.apply_translation(-normalized.bounding_box.centroid)
        normalized.apply_scale(1.0 / scale)
        return normalized
