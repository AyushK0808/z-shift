"""B13 - a second reconstruction backend through the same artifacts. Needs DUSt3R and a GPU.

Answers the reviewer's request to demonstrate stage replaceability. The
reconstruction stage is run with a different backend (DUSt3R, by default) on
the five DTU scans B2 uses, at the same 10 spread views with complete
pairing, scored by the same DTU protocol and the same camera fit. Nothing
downstream is changed to accommodate it, and each row records that:

- `contract_ok`: the stage wrote its mesh, `point_cloud.ply`, `cameras.json`
  and `run_manifest.json`, and the manifest names the backend that ran;
- `refine_ok`: refinement consumed the mesh through the unchanged `clean_mesh`.

The MASt3R rows are B2's own 10-view rows (same views, seed and scoring); this
experiment adds the other backend so the two sit side by side. DUSt3R is not
metric, so its scale comes from the camera fit (`camera_align_scale`), and
`mast3r_scale_ratio` is meaningless for it.

    uv run python -m bench.exp_b13_backend_swap --manifest ~/zshift/dtu/manifest.json
"""

from __future__ import annotations

import logging
from pathlib import Path

from bench.csvio import ResultWriter
from bench.exp_b2_reconstruction_accuracy import _out_of_memory_errors, _release_gpu_memory
from bench.harness import experiment_parser, finish
from bench.tier_b_common import (
    SceneSet,
    bounded_pairing,
    build_job,
    iter_scenes,
    load_gt_points,
    point_cloud_from_output,
    reconstruction_scale_for,
    require_cuda_device,
    reset_gpu_peak,
    run_reconstruction,
    score_against_gt,
)
from spatial_ingestion.reconstruction.backends import available_backends, get_backend
from spatial_ingestion.reconstruction.models import Mast3rRunParams

EXP_ID = "b13_backend_swap"
N_IMAGES = 10
REQUIRED_ARTIFACTS = ("point_cloud.ply", "cameras.json", "run_manifest.json")

logger = logging.getLogger(__name__)


def _refine(mesh_path: Path) -> dict[str, object]:
    """Hand the stage's mesh to refinement exactly as the pipeline does."""
    from spatial_ingestion.refinement import MeshCleaningConfig, clean_mesh, load_mesh_file

    try:
        result = clean_mesh(
            load_mesh_file(mesh_path), MeshCleaningConfig(mode="object", smoothing_iters=15)
        )
    except Exception as exc:  # noqa: BLE001 - a refusal is a result
        return {"refine_ok": False, "refine_error": f"{type(exc).__name__}: {exc}"[:300]}
    return {
        "refine_ok": True,
        "refine_in_cells": int(result["input_cell_count"]),
        "refine_out_cells": int(result["output_cell_count"]),
    }


def run(
    results_dir: Path | None = None,
    *,
    manifest: Path,
    backends: tuple[str, ...] = ("dust3r",),
    scene_names: list[str] | None = None,
    seed: int = 0,
    image_size: int = 512,
    output_root: Path | None = None,
    quick: bool = False,
) -> ResultWriter:
    writer = ResultWriter(EXP_ID, results_dir)
    scenes = iter_scenes(SceneSet.from_manifest(manifest), scene_names)
    if quick:
        scenes = scenes[:1]
    root = output_root or (Path("data") / "bench" / EXP_ID)
    device = require_cuda_device()

    for scene in scenes:
        if scene.gt_path is None:
            logger.warning("scene %s has no ground truth; skipping", scene.name)
            continue
        gt_points = load_gt_points(scene.gt_path, seed=seed)
        image_paths = scene.image_paths(N_IMAGES, spread=True)
        for name in backends:
            backend = get_backend(name)
            params = Mast3rRunParams(
                model_name=backend.default_model,
                image_size=image_size,
                seed=seed,
                device=device,
                pairing_strategy=bounded_pairing(len(image_paths), 800, cyclic=False),
            )
            base_row = {
                "scene": scene.name,
                "backend": name,
                "model_name": params.model_name,
                "n_images": len(image_paths),
                "view_selection": "spread",
                "image_size": image_size,
                "pairing_strategy": params.pairing_strategy,
            }
            output_path = root / name / scene.name / f"{scene.name}.glb"
            job = build_job(image_paths, output_path, params=params, label=scene.name, backend=name)
            try:
                reset_gpu_peak()
                run_info = run_reconstruction(job)
            except _out_of_memory_errors() as exc:
                logger.error("%s/%s: %s", name, scene.name, exc)
                writer.add(**base_row, status=f"{type(exc).__name__}: {exc}"[:300])
                _release_gpu_memory()
                continue

            output_dir = run_info["output_dir"]
            recorded = run_info["manifest"].get("backend")
            contract_ok = (
                all((output_dir / artifact).exists() for artifact in REQUIRED_ARTIFACTS)
                and Path(run_info["output_path"]).exists()
                and recorded == name
            )
            outputs = run_info["manifest"].get("outputs", {})
            scores = score_against_gt(
                point_cloud_from_output(output_dir, seed=seed),
                gt_points,
                tau=scene.tau,
                seed=seed,
                reconstruction_scale=reconstruction_scale_for(scene.units),
                reference=scene.dtu_reference(),
                output_dir=output_dir,
            )
            writer.add(
                **base_row,
                status="ok",
                contract_ok=contract_ok,
                manifest_backend=recorded,
                wall_seconds=run_info["wall_seconds"],
                gpu_peak_mb=run_info["gpu_peak_mb"],
                **{f"out_{key}": value for key, value in outputs.items()},
                **{
                    f"stage_{record['stage']}_s": record["seconds"]
                    for record in run_info["manifest"].get("stage_timings", [])
                },
                **_refine(Path(run_info["output_path"])),
                **scores,
            )
            logger.info(
                "%s/%s: contract_ok=%s F=%.3f overall=%.2f",
                name,
                scene.name,
                contract_ok,
                scores["f_score"],
                scores["dtu_overall"],
            )
            _release_gpu_memory()
    return writer


def main(argv: list[str] | None = None) -> int:
    parser = experiment_parser(EXP_ID, __doc__ or "")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=available_backends(),
        default=["dust3r"],
        help="backends to run; the MASt3R rows are B2's 10-view rows",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    finish(
        run(
            args.results_dir,
            manifest=args.manifest,
            backends=tuple(args.backends),
            scene_names=args.scenes,
            seed=args.seed,
            image_size=args.image_size,
            output_root=args.output_root,
            quick=args.quick,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
