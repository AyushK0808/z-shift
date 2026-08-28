"""B1 - end-to-end functional validation on real captures. Requires MASt3R + GPU.

Replaces the current evidence for SV-A, which is a monkeypatched `pv.Sphere()`
in every end-to-end test plus one skipped real-data test that runs Phase 2+3
only, with smoothing off. A reviewer who opens the test suite will find the
sphere; SV-A should say plainly that CI coverage is contract-level and that
end-to-end validation was performed manually on N captures, with this table
behind it.

Runs the real CLI path for each source type (image folder, single video, video
folder) and checks off every artifact the pipeline promises. The rigged GLB
still needs a human to open it in Blender and confirm the armature imports with
bones bound; that column is recorded as a manual check, not asserted here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from bench.csvio import ResultWriter
from bench.harness import experiment_parser, finish
from bench.instrument import peak_rss_mb
from bench.tier_b_common import SceneSet, clear_alignment_cache, iter_scenes, timed
from spatial_ingestion.auto_rigging.models import ArticulationType, AutoRigConfig
from spatial_ingestion.final_pipeline.handoff import run_ingested_pipeline
from spatial_ingestion.outcomes_engine.engine import DEFAULT_DELIVERABLES_ROOT
from spatial_ingestion.reconstruction.models import Mast3rRunParams
from spatial_ingestion.refinement import MeshCleaningConfig

EXP_ID = "b1_end_to_end"

# Every artifact SV-A claims the pipeline produces. Present/absent per run.
EXPECTED_ARTIFACTS: tuple[str, ...] = (
    "raw_mesh_path",
    "refined_mesh_path",
    "reconstruction_manifest_path",
    "refinement_manifest_path",
)
RIG_ARTIFACTS: tuple[str, ...] = (
    "rigged_mesh_path",
    "rigging_manifest_path",
    "skeleton_path",
    "skinning_weights_path",
)

logger = logging.getLogger(__name__)


def _artifact_row(result: Any, rig: bool) -> dict[str, Any]:
    pipeline_result = getattr(result, "pipeline_result", result)
    row: dict[str, Any] = {}
    wanted = EXPECTED_ARTIFACTS + (RIG_ARTIFACTS if rig else ())
    for name in wanted:
        path = getattr(pipeline_result, name, None)
        row[f"has_{name}"] = bool(path and Path(path).exists())
        row[f"bytes_{name}"] = Path(path).stat().st_size if path and Path(path).exists() else 0

    point_cloud = Path(pipeline_result.reconstruction_manifest_path).parent / "point_cloud.ply"
    row["has_point_cloud"] = point_cloud.exists()
    row["bytes_point_cloud"] = point_cloud.stat().st_size if point_cloud.exists() else 0

    diagnostics = pipeline_result.refinement_diagnostics or {}
    row.update(
        {
            "refined_is_watertight": diagnostics.get("is_watertight"),
            "refined_open_edges": diagnostics.get("open_edge_count"),
            "refined_out_cells": diagnostics.get("output_cell_count"),
            "refined_in_cells": diagnostics.get("input_cell_count"),
            "refinement_warnings": "; ".join(diagnostics.get("warnings", []))[:300],
        }
    )
    deliverable = getattr(result, "deliverable", None)
    if deliverable is not None:
        row["deliverable_path"] = deliverable.output_path
        row["deliverable_track"] = deliverable.track
        row["has_deliverable"] = bool(
            deliverable.output_path and Path(deliverable.output_path).exists()
        )
    return row


def run(
    results_dir: Path | None = None,
    *,
    manifest: Path,
    scene_names: list[str] | None = None,
    seed: int = 0,
    image_size: int = 512,
    tsdf_thresh: float = 0.2,
    use_cases: tuple[str, ...] = ("editing", "viewing"),
    rig: bool = True,
    articulation: ArticulationType = ArticulationType.BIPED,
    n_images: int | None = None,
    deliverables_root: Path | None = None,
) -> ResultWriter:
    writer = ResultWriter(EXP_ID, results_dir)
    scenes = iter_scenes(SceneSet.from_manifest(manifest), scene_names)

    for scene in scenes:
        inputs = scene.image_paths(limit=n_images)
        for use_case in use_cases:
            # Rigging only applies to the editing track; run_full_pipeline
            # rejects --rig for viewing, which is itself part of SIV-E.
            do_rig = rig and use_case == "editing"
            row: dict[str, Any] = {
                "scene": scene.name,
                "n_inputs": len(inputs),
                "use_case": use_case,
                "rig_requested": do_rig,
                "articulation": articulation.value if do_rig else "",
                "image_size": image_size,
                "tsdf_thresh": tsdf_thresh,
                "seed": seed,
                "blender_armature_check": "manual: not asserted here",
            }
            try:
                with timed(f"{scene.name}/{use_case}") as timer:
                    result = run_ingested_pipeline(
                        inputs,
                        use_case=use_case,
                        mast3r_params=Mast3rRunParams(
                            image_size=image_size, tsdf_thresh=tsdf_thresh, seed=seed
                        ),
                        refinement_config=MeshCleaningConfig(mode="object"),
                        rigging_config=(
                            AutoRigConfig(articulation_type=articulation) if do_rig else None
                        ),
                        deliverables_root=deliverables_root or DEFAULT_DELIVERABLES_ROOT,
                    )
                row.update(
                    status="ok",
                    wall_seconds=timer.seconds,
                    peak_rss_mb=peak_rss_mb(),
                    **_artifact_row(result, do_rig),
                )
            except Exception as exc:  # noqa: BLE001 - a failed capture is a result
                row.update(status=f"{type(exc).__name__}: {exc}"[:300])
                logger.exception("%s/%s failed", scene.name, use_case)
            writer.add(**row)

            pipeline_result = getattr(locals().get("result", None), "pipeline_result", None)
            if pipeline_result is not None:
                clear_alignment_cache(Path(pipeline_result.reconstruction_manifest_path).parent)
    return writer


def main(argv: list[str] | None = None) -> int:
    parser = experiment_parser(EXP_ID, __doc__ or "")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--tsdf-thresh", type=float, default=0.2)
    parser.add_argument("--no-rig", action="store_true")
    parser.add_argument(
        "--articulation",
        choices=[kind.value for kind in ArticulationType],
        default=ArticulationType.BIPED.value,
        help="skeleton template to fit; use 'static' for object captures",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=None,
        help="cap inputs per scene; without it a 313-frame capture runs in full",
    )
    parser.add_argument("--deliverables-root", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    writer = run(
        results_dir=args.results_dir,
        manifest=args.manifest,
        scene_names=args.scenes,
        seed=args.seed,
        image_size=args.image_size,
        tsdf_thresh=args.tsdf_thresh,
        rig=not args.no_rig,
        articulation=ArticulationType(args.articulation),
        n_images=args.n_images,
        deliverables_root=args.deliverables_root,
    )
    finish(writer)
    for row in writer.rows:
        print(
            f"  {row['scene']:<14}{row['use_case']:<9}{row['status']:<12}"
            f"{row.get('wall_seconds', 0):>8.1f}s  watertight={row.get('refined_is_watertight')}"
            f"  rigged_glb={row.get('has_rigged_mesh_path', False)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
