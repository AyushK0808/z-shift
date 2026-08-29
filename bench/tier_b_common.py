"""Shared plumbing for Tier B, which needs MASt3R weights and a GPU.

Nothing in this module runs during Tier A. It is imported lazily by the
exp_b*.py modules so that importing `bench` on a CPU-only laptop never pulls in
torch weights or dust3r.

Scene loading, GT loading and the reconstruct-and-score loop are factored out
here because B2, B3, B4, B6 and B8 differ only in which knob they turn.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import re
import shutil
import tempfile
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bench.gt_align import align_to_reference
from bench.instrument import peak_rss_mb
from bench.metrics import chamfer_l1, precision_recall_f, sample_points
from spatial_ingestion.instrumentation import StageLog
from spatial_ingestion.reconstruction.models import (
    Mast3rRunParams,
    ReconstructionJob,
    ReconstructionMode,
)

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# A suffix filter cannot tell a photograph from a depth map -- .png is .png.
# NeRF-synthetic ships r_0_depth_0000.png and r_0_normal_0000.png beside the RGB
# frames; handing those to MASt3R as photographs corrupts the reconstruction
# without failing anything, so they are excluded by filename and by folder.
NON_PHOTO_RE = re.compile(
    r"(?:^|[_-])(depth|normals?|masks?|alpha|segs?|segmentation)(?:[_-]|\d|$)",
    re.IGNORECASE,
)
NON_PHOTO_DIRS = frozenset(
    {"depth", "depths", "normal", "normals", "mask", "masks", "alpha", "seg", "segmentation"}
)


def is_photo(path: Path) -> bool:
    """True for a file that is a photograph rather than a rendered map."""
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return False
    if NON_PHOTO_RE.search(path.stem):
        return False
    return path.parent.name.lower() not in NON_PHOTO_DIRS


def natural_key(path: Path) -> tuple[object, ...]:
    """Sort key that reads digit runs as numbers.

    Every B module treats this ordering as capture order. A plain lexical sort
    breaks it on unpadded numbering -- r_109.png sorts before r_11.png -- which
    scrambles the trajectory that B3's windowed pairing and B4's frame budget
    both assume is sequential. re.split with a capture group always alternates
    non-digit, digit, so positions stay type-consistent and comparable.
    """
    return tuple(
        int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)
    )


# DTU's own evaluation convention. Restate tau on every row that reports an
# F-score; a bare F-score is not interpretable.
DTU_TAU_MM = 2.0

# The metric MASt3R checkpoint (*_catmlpdpt_metric) predicts real-world scale
# in metres. Nothing in export_scene_to_mesh / point_cloud_from_output converts
# that, so scoring it against a ground truth stated in another unit silently
# compares a cloud that is 1000x too small: a rigid (no-scale) ICP cannot make
# up a 1000x size difference, so it collapses the whole reconstruction onto
# whichever small pocket of the GT surface is nearest, which reads as
# precision~1.0, recall~0 and a deceptively tiny align_rmse -- not as a
# failure. This factor is applied by `score_against_gt` alone, so it affects
# only benchmark scoring; the actual `.glb` / `point_cloud.ply` the pipeline
# writes stays in MASt3R's native metres.
MAST3R_METRIC_UNITS = "m"


def reconstruction_scale_for(units: str) -> float:
    """Factor converting MASt3R's native metric output into `units`.

    Only "mm" gets a real conversion. Anything else -- COLMAP's arbitrary SfM
    scale, "unknown (auto-discovered)" -- has no fixed real-world relationship
    to metres, so guessing a factor there would trade one silent scale bug for
    another; run those with `with_scale=True` instead.
    """
    return 1000.0 if units == "mm" else 1.0


@dataclass
class Scene:
    """One capture plus, optionally, its ground truth."""

    name: str
    image_dir: Path
    gt_path: Path | None = None
    tau: float = DTU_TAU_MM
    units: str = "mm"
    notes: str = ""
    # Optional glob selecting which files in image_dir form the capture. DTU's
    # Rectified/scanNN holds every viewpoint under 7 lighting conditions
    # (rect_001_0_r5000.png .. rect_001_6_r5000.png); taking the directory
    # whole feeds MASt3R seven near-duplicate copies of each pose, which
    # inflates the pair count and tells the reconstruction nothing new.
    image_glob: str | None = None

    def image_paths(self, limit: int | None = None) -> list[Path]:
        candidates = (
            self.image_dir.glob(self.image_glob) if self.image_glob else self.image_dir.iterdir()
        )
        paths = sorted((path for path in candidates if is_photo(path)), key=natural_key)
        if not paths:
            detail = f" matching {self.image_glob}" if self.image_glob else ""
            raise FileNotFoundError(f"no images under {self.image_dir}{detail}")
        return paths[:limit] if limit else paths


@dataclass
class SceneSet:
    """A Tier B dataset described by a small JSON manifest.

    Keeping the dataset out of the code means B1-B8 do not hardcode DTU paths,
    and a reviewer can point the same experiments at Tanks and Temples or at a
    local capture with COLMAP pseudo-GT by editing one file.
    """

    scenes: list[Scene] = field(default_factory=list)

    @classmethod
    def from_manifest(cls, path: Path | str) -> SceneSet:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        root = Path(path).resolve().parent
        scenes = []
        for entry in data["scenes"]:
            scenes.append(
                Scene(
                    name=entry["name"],
                    image_dir=(root / entry["image_dir"]).resolve(),
                    gt_path=((root / entry["gt_path"]).resolve() if entry.get("gt_path") else None),
                    tau=float(entry.get("tau", data.get("tau", DTU_TAU_MM))),
                    units=entry.get("units", data.get("units", "mm")),
                    notes=entry.get("notes", ""),
                    image_glob=entry.get("image_glob") or data.get("image_glob"),
                )
            )
        return cls(scenes=scenes)


# B2, B3, B5 and B6 each run as their own subprocess (section 13 of the Tier B
# notebook), and B6 additionally loads its GT file a second time for face
# connectivity -- so with no cache, DTU's raw structured-light scans (tens of
# millions of points before subsampling) get trimesh.load()'d from scratch up
# to five times per scene for a byte-identical result every time (same file,
# same seed). Cached to disk, keyed on the source file's own size/mtime so a
# changed mount can't serve a stale array; ZSHIFT_GT_CACHE_DIR overrides the
# location, e.g. to survive across Kaggle sessions on a persisted output
# volume.
_GT_CACHE_DIR = Path(
    os.environ.get("ZSHIFT_GT_CACHE_DIR", Path(tempfile.gettempdir()) / "zshift_gt_cache")
)


def _gt_cache_key(resolved: Path, *parts: object) -> Path:
    stat = resolved.stat()
    digest = hashlib.sha256(
        "|".join(
            [str(resolved), str(stat.st_size), str(stat.st_mtime_ns), *map(str, parts)]
        ).encode()
    ).hexdigest()
    return _GT_CACHE_DIR / digest


def load_gt_object(path: Path) -> Any:
    """Load a GT point cloud or mesh via trimesh, cached once per source file.

    `load_gt_points` only needs points, but B6 separately needs face
    connectivity (for `normal_consistency`) and used to call `trimesh.load`
    on the same GT file a second time, uncached, on top of whatever
    `load_gt_points` was already doing. Both now come through here. Pickled
    rather than re-exported, so a cache hit skips trimesh's own parse/validate
    pass entirely instead of just moving where it happens.
    """
    resolved = Path(path).resolve()
    cache_path = _gt_cache_key(resolved, "raw").with_suffix(".pkl")
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            return pickle.load(handle)  # noqa: S301 - our own cache, not external input

    import trimesh

    loaded = trimesh.load(str(resolved))
    _GT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(loaded, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return loaded


def load_gt_points(path: Path, max_points: int = 500_000, seed: int = 0) -> np.ndarray:
    """Load a GT point cloud or mesh as a point array, subsampled for tractability."""
    import trimesh

    resolved = Path(path).resolve()
    cache_path = _gt_cache_key(resolved, max_points, seed).with_suffix(".npy")
    if cache_path.exists():
        return np.load(cache_path)

    loaded = load_gt_object(resolved)
    if isinstance(loaded, trimesh.PointCloud):
        points = np.asarray(loaded.vertices, dtype=float)
    elif isinstance(loaded, trimesh.Trimesh):
        points = sample_points(loaded, min(max_points, 500_000), seed)
    else:
        raise ValueError(f"unsupported ground-truth artifact: {path}")

    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(len(points), max_points, replace=False)]

    _GT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, points)
    return points


def clear_alignment_cache(output_dir: Path) -> None:
    """Delete the sparse-alignment cache.

    `run_sparse_alignment` reuses `<output_dir>/cache`, so without this a repeat
    run silently replays the first one and any determinism or ablation result is
    vacuous. Called before every independent run in Tier B.
    """
    cache = output_dir / "cache"
    if cache.exists():
        shutil.rmtree(cache)
        logger.info("cleared alignment cache: %s", cache)


def build_job(
    image_paths: Sequence[Path],
    output_path: Path,
    *,
    params: Mast3rRunParams,
    mode: ReconstructionMode = ReconstructionMode.MULTI_VIEW,
    label: str = "tier_b",
) -> ReconstructionJob:
    return ReconstructionJob(
        mode=mode,
        image_uris=[path.resolve().as_uri() for path in image_paths],
        label=label,
        output_path=str(output_path.resolve()),
        metadata=params.model_dump(),
    )


def run_reconstruction(job: ReconstructionJob, *, clear_cache: bool = True) -> dict[str, Any]:
    """Run Phase 2 and return the manifest plus wall-clock and memory."""
    from spatial_ingestion.reconstruction.pipeline import _resolve_output_paths
    from spatial_ingestion.reconstruction.pipeline import run as run_phase2

    output_path, output_dir = _resolve_output_paths(job)
    output_dir.mkdir(parents=True, exist_ok=True)
    if clear_cache:
        clear_alignment_cache(output_dir)

    stage_log = StageLog()
    with stage_log.stage("phase2_total"):
        exit_code = run_phase2(job)

    manifest_path = output_dir / "run_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    return {
        "exit_code": exit_code,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "output_path": output_path,
        "output_dir": output_dir,
        "wall_seconds": stage_log.total_seconds,
        "peak_rss_mb": peak_rss_mb(),
        "gpu_peak_mb": gpu_peak_mb(),
    }


def require_cuda_device() -> str:
    """Resolve the device Tier B must run on: CUDA, no fallback.

    Tier B reports wall-clock time and peak memory. A silent CPU or MPS
    fallback (what `Mast3rRunParams(device="auto")` resolves to when CUDA is
    missing) would produce numbers that look like results but measure the
    wrong hardware, so B1-B6 and B8 call this instead of accepting "auto".
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Tier B benchmarks require a CUDA GPU. torch.cuda.is_available() "
            "is False on this machine -- run on a CUDA-enabled host, or use "
            "the Phase 2 CLI directly (spatial_ingestion.reconstruction.cli) "
            "with --device cpu for non-benchmark reconstruction."
        )
    return "cuda"


def gpu_peak_mb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    except Exception:  # noqa: BLE001
        return None


def reset_gpu_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception as exc:  # noqa: BLE001 - resetting a counter must never
        # take down a benchmark run; a missing GPU stat is a blank cell.
        logger.debug("could not reset GPU peak stats: %s", exc)


def score_against_gt(
    reconstruction_points: np.ndarray,
    gt_points: np.ndarray,
    *,
    tau: float,
    with_scale: bool = False,
    seed: int = 0,
    reconstruction_scale: float = 1.0,
) -> dict[str, Any]:
    """Align to GT, then report accuracy/completeness/F-score and the residual.

    The alignment residual is reported alongside the metrics, not instead of
    them: a good Chamfer after a badly-fitted alignment means nothing.

    `reconstruction_scale` converts MASt3R's native metric (metre) output into
    the ground truth's units before alignment -- see `reconstruction_scale_for`.
    `with_scale`'s ICP fit cannot be trusted to recover this on its own: with
    unknown correspondences it can "solve" a real scale mismatch by shrinking
    to a point instead of finding the true ratio.
    """
    reconstruction_points = np.asarray(reconstruction_points, dtype=float) * reconstruction_scale
    aligned, alignment = align_to_reference(reconstruction_points, gt_points, with_scale=with_scale)
    precision, recall, f_score = precision_recall_f(aligned, gt_points, tau)
    from bench.metrics import _dists  # noqa: PLC0415 - internal helper, one call site

    return {
        "tau": tau,
        "reconstruction_scale": reconstruction_scale,
        "chamfer_l1": round(chamfer_l1(aligned, gt_points), 6),
        "accuracy_mean": round(float(_dists(aligned, gt_points).mean()), 6),
        "completeness_mean": round(float(_dists(gt_points, aligned).mean()), 6),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f_score": round(f_score, 4),
        "n_reconstruction_points": int(len(aligned)),
        "n_gt_points": int(len(gt_points)),
        "seed": seed,
        **alignment.as_row(),
    }


def point_cloud_from_output(output_dir: Path, max_points: int = 500_000, seed: int = 0):
    """Read the point_cloud.ply Phase 2 writes next to its mesh."""
    return load_gt_points(output_dir / "point_cloud.ply", max_points=max_points, seed=seed)


def timed(label: str) -> Any:
    """Small helper so B-modules can time an inline block without a StageLog."""

    class _Timer:
        def __enter__(self) -> _Timer:
            self.start = time.perf_counter()
            return self

        def __exit__(self, *exc: object) -> None:
            self.seconds = round(time.perf_counter() - self.start, 4)
            logger.info("%s: %.2f s", label, self.seconds)

    return _Timer()


def iter_scenes(scene_set: SceneSet, names: Iterable[str] | None = None) -> list[Scene]:
    if names is None:
        return list(scene_set.scenes)
    wanted = set(names)
    return [scene for scene in scene_set.scenes if scene.name in wanted]
