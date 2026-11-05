"""B11 - CO3D teddybear reconstruction scored against CO3D's reference cloud. Needs B9's outputs.

Answers the reviewer's request for CO3D accuracy. The reference is CO3D's own
`pointcloud.ply`, a COLMAP reconstruction: a pseudo-reference, registered to by
ICP because the single-sequence chunk ships no camera annotations (see
bench/co3d_eval.py for what that does and does not support). The paper must
say "agreement with CO3D's reference", not "ground-truth accuracy".

For each B9 run directory (`--runs label=path`), the raw mesh is refined with
object isolation on, so the registration sees the subject without the floor the
reference lacks. Rows score three things under that one registration: the
isolated refined mesh, the refinement the run shipped with, and the raw point
cloud inside the reference's padded bounding box.

    uv run python -m bench.exp_b11_co3d_accuracy \\
        --data-dir ~/zshift/co3d/teddybear \\
        --runs views8=~/zshift/case_study/run views24=~/zshift/case_study_24/run
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from bench.co3d_eval import CO3D_TEDDYBEAR_URL, Co3dReference, fetch_reference, register, score_co3d
from bench.csvio import ResultWriter
from bench.exp_b9_case_study import DEFAULT_SEQUENCE
from bench.harness import experiment_parser, finish
from bench.metrics import sample_points
from bench.tier_b_common import load_gt_points

EXP_ID = "b11_co3d_accuracy"
SAMPLE_POINTS = 100_000

logger = logging.getLogger(__name__)


def _mesh_samples(mesh: Any, seed: int) -> np.ndarray:
    from spatial_ingestion.refinement import to_trimesh

    tri = to_trimesh(mesh)
    return sample_points(tri, SAMPLE_POINTS, seed) if len(tri.faces) else np.asarray(tri.vertices)


def run(
    results_dir: Path | None = None,
    *,
    data_dir: Path,
    runs: dict[str, Path],
    sequence: str = DEFAULT_SEQUENCE,
    seed: int = 0,
    url: str = CO3D_TEDDYBEAR_URL,
) -> ResultWriter:
    from spatial_ingestion.refinement import MeshCleaningConfig, clean_mesh, load_mesh_file

    writer = ResultWriter(EXP_ID, results_dir)
    reference_path = fetch_reference(data_dir, sequence, url)
    reference = Co3dReference.from_ply(reference_path, seed=seed)
    logger.info("reference: %d points, diagonal %.4g", len(reference.points), reference.diagonal)

    for label, run_dir in runs.items():
        raw_path = run_dir / "mesh.glb"
        if not raw_path.exists():
            logger.warning("%s: no raw mesh at %s; skipping", label, raw_path)
            continue
        raw_poly = load_mesh_file(raw_path)
        isolated = clean_mesh(
            raw_poly,
            MeshCleaningConfig(mode="object", smoothing_iters=15, isolate_object=True),
        )
        isolated_points = _mesh_samples(isolated["mesh"], seed)
        alignment, registration = register(isolated_points, reference, seed)
        base = {
            "run": label,
            "sequence": sequence,
            "n_reference_points": len(reference.points),
            "reference_diagonal": round(reference.diagonal, 5),
            "support_plane_removed": isolated.get("support_plane_removed", ""),
            "reference_kind": "colmap_pseudo_reference",
            "alignment": "rms_scale_then_icp",
            **registration,
        }
        variants: dict[str, np.ndarray] = {"refined_isolated": isolated_points}
        shipped = run_dir / "mesh_refined.glb"
        if shipped.exists():
            variants["refined_as_shipped"] = _mesh_samples(load_mesh_file(shipped), seed)
        cloud = run_dir / "point_cloud.ply"
        if cloud.exists():
            variants["raw_point_cloud"] = load_gt_points(cloud, max_points=500_000, seed=seed)
        for variant, points in variants.items():
            writer.add(**base, variant=variant, **score_co3d(points, reference, alignment))
            logger.info("%s %s: scored", label, variant)
    return writer


def _parse_runs(items: list[str]) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for item in items:
        label, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--runs expects label=path, got {item!r}")
        runs[label] = Path(path).expanduser()
    return runs


def main(argv: list[str] | None = None) -> int:
    parser = experiment_parser(EXP_ID, __doc__ or "")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    parser.add_argument("--runs", nargs="+", required=True, help="label=path of B9 run directories")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    finish(
        run(
            args.results_dir,
            data_dir=args.data_dir.expanduser(),
            runs=_parse_runs(args.runs),
            sequence=args.sequence,
            seed=args.seed,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
