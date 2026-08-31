"""Colab end-to-end demo: CO3D teddybear -> MASt3R -> refine -> geometry-guided rig.

Runs the full z-shift chain on a Google Colab session using the repo checked
out at ``ZSHIFT_REPO`` (default ``/content/z-shift``) and the CO3D
``teddybear_001_singlesequence`` chunk (the ``_000`` chunk is annotation-only
-- no images). Use with the Google Colab CLI::

    colab upload <repo>/zshift_code.tar /content/
    colab exec -s zshift --file scripts/colab_teddybear_demo.py

The script is idempotent: dependency install, MASt3R setup, and the ~606 MB
CO3D download all skip once present, and ``--rig-only`` re-rigs the cached
refined mesh without another expensive reconstruction.

Flags (argv, before anything else):
    --rig-only          skip reconstruction/refinement and re-rig the cached mesh
    --num-frames N      frames fed to MASt3R after even subsampling (default 28)
    --articulation A    biped | quadruped | winged | static (default biped)
    --resolution R      voxel grid resolution for skinning (default 64)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig-only", action="store_true")
    parser.add_argument("--num-frames", type=int, default=28)
    parser.add_argument("--articulation", default="biped")
    parser.add_argument("--resolution", type=int, default=64)
    return parser.parse_args()


ARGS = parse_args()

BASE_DIR = Path(os.environ.get("ZSHIFT_BASE", "/content"))
REPO_DIR = Path(os.environ.get("ZSHIFT_REPO", str(BASE_DIR / "z-shift")))
WORK_DIR = BASE_DIR / "work"
OUT_DIR = BASE_DIR / "out"
CO3D_ZIP_URL = os.environ.get(
    "CO3D_ZIP_URL",
    "https://dl.fbaipublicfiles.com/co3dv2_231130/teddybear_001_singlesequence.zip",
)
PINNED_MAST3R = "f5209afc300cec36239a7ac992263f36847bbba0"

for d in (WORK_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

print("=== 1. GPU check ===")
import torch

print(f"torch {torch.__version__} | cuda build {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU required")
props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name}, {props.total_memory / 1024**3:.1f} GB")

for p in (str(REPO_DIR), str(REPO_DIR / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
for p in (
    str(REPO_DIR / "third_party" / "mast3r"),
    str(REPO_DIR / "third_party" / "mast3r" / "dust3r"),
):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["PYVISTA_OFF_SCREEN"] = "true"


print("=== 2. Dependencies (idempotent) ===")
import tomllib

with open(REPO_DIR / "pyproject.toml", "rb") as fh:
    pyproject = tomllib.load(fh)
SKIP_PREFIXES = ("torch", "torchvision", "numpy")
deps = []
for dep in pyproject["project"]["dependencies"]:
    name = dep.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip().lower()
    if name.startswith(SKIP_PREFIXES):
        continue
    deps.append(dep)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *deps], check=True)  # noqa: S603 -- trusted pinned deps
print(f"deps ensured ({len(deps)})")

print("=== 3. CO3D teddybear (chunk 001 = real images) ===")
zip_path = WORK_DIR / "teddybear_001_singlesequence.zip"
extract_dir = WORK_DIR / "teddybear_extracted"
if not zip_path.exists():
    print(f"downloading {CO3D_ZIP_URL} (606 MB, one-time)")
    urllib.request.urlretrieve(CO3D_ZIP_URL, zip_path)  # noqa: S310 -- pinned https dataset URL
print(f"zip: {zip_path.stat().st_size / 1024**2:.1f} MB")


# Extract fresh unless a previous extraction already has real images -- the old
# ``_000`` chunk left an annotation-only directory behind, which aborts the
# sequence search if we blindly reuse it.
def _has_images(root: Path) -> bool:
    return any(p.is_dir() for p in root.rglob("images"))


if not extract_dir.exists() or not _has_images(extract_dir):
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    print("extracted fresh into", extract_dir)

image_dirs = sorted(p for p in extract_dir.rglob("images") if p.is_dir())
if not image_dirs:
    raise SystemExit("no images found in CO3D chunk -- wrong URL or partial download")
SEQUENCE_DIR = max(image_dirs, key=lambda d: sum(1 for _ in d.iterdir()))
all_frames = sorted(SEQUENCE_DIR.glob("*.jpg")) or sorted(SEQUENCE_DIR.glob("*.png"))
print(f"{len(all_frames)} frames in {SEQUENCE_DIR.relative_to(extract_dir)}")

SUBSET_DIR = WORK_DIR / "teddybear_subset"
if not ARGS.rig_only:
    if SUBSET_DIR.exists():
        shutil.rmtree(SUBSET_DIR)
    SUBSET_DIR.mkdir(parents=True)
    indices = sorted(set(np.linspace(0, len(all_frames) - 1, ARGS.num_frames).astype(int).tolist()))
    for i, idx in enumerate(indices):
        shutil.copy(all_frames[idx], SUBSET_DIR / f"frame_{i:03d}{all_frames[idx].suffix}")
    print(f"curated {len(indices)} frames; complete pairing -> C({len(indices)},2) pairs")

from spatial_ingestion.auto_rigging.models import ArticulationType, AutoRigConfig  # noqa: E402
from spatial_ingestion.final_pipeline.handoff import run_ingested_pipeline  # noqa: E402
from spatial_ingestion.reconstruction.cli import DEFAULT_MODEL, collect_input_images  # noqa: E402
from spatial_ingestion.reconstruction.models import Mast3rRunParams  # noqa: E402
from spatial_ingestion.refinement import MeshCleaningConfig  # noqa: E402

RAW_MESH_PATH = OUT_DIR / "mesh.glb"
REFINED_MESH_PATH = OUT_DIR / "mesh_refined.glb"
RIG_DIR = OUT_DIR / "rig"
RIGGED_GLB_PATH = OUT_DIR / "rigged_mesh.glb"
DELIVERABLES_DIR = OUT_DIR / "deliverables"
INPUT_TYPE = "image_folder"

result = None
if ARGS.rig_only:
    print("=== 4. rig-only: loading cached refined mesh ===")
    from spatial_ingestion.auto_rigging.export import RigMetadataExporter
    from spatial_ingestion.auto_rigging.pipeline import AutoRiggingPipeline

    if not REFINED_MESH_PATH.exists():
        raise SystemExit(f"no cached refined mesh at {REFINED_MESH_PATH}")
    export_cfg = AutoRigConfig(
        articulation_type=ArticulationType(ARGS.articulation),
        max_skinning_influences=4,
        detailed_skeleton=True,
        normalize_mesh=True,
        output_dir=RIG_DIR,
        rigged_output_path=RIGGED_GLB_PATH,
    )
    rigging_result = AutoRiggingPipeline(exporter=RigMetadataExporter(RIG_DIR)).rig_mesh_file(
        REFINED_MESH_PATH, config=export_cfg
    )
    skeleton_path = Path(rigging_result.skeleton_uri.replace("file://", ""))
    weights_path = Path(rigging_result.weights_uri.replace("file://", ""))
    result_refined_diags = {}
else:
    print("=== 4. full pipeline: ingest -> MASt3R -> refine -> rig ===")
    mast3r_params = Mast3rRunParams(
        model_name=DEFAULT_MODEL,
        device="cuda",
        image_size=512,
        pairing_strategy="complete",
        tsdf_thresh=0.2,
        min_conf_thr=1.5,
        seed=42,
        deterministic=True,
        dry_run=False,
    )
    refinement_config = MeshCleaningConfig(
        mode="object",
        smoothing_iters=15,
        pass_band=0.1,
        decimate_target_reduction=None,
        verify_watertight=True,
    )
    rigging_config = AutoRigConfig(
        articulation_type=ArticulationType(ARGS.articulation),
        max_skinning_influences=4,
        detailed_skeleton=True,
        normalize_mesh=True,
        output_dir=RIG_DIR,
        rigged_output_path=RIGGED_GLB_PATH,
    )
    image_paths = collect_input_images(SUBSET_DIR)
    t0 = time.time()
    full_result = run_ingested_pipeline(
        image_paths,
        mast3r_params=mast3r_params,
        output_path=RAW_MESH_PATH,
        use_case="editing",
        input_type=INPUT_TYPE,
        refinement_config=refinement_config,
        refined_output_path=REFINED_MESH_PATH,
        rig=True,
        rigging_config=rigging_config,
        rigged_output_path=RIGGED_GLB_PATH,
        rig_output_dir=RIG_DIR,
        deliverables_root=DELIVERABLES_DIR,
    )
    print(f"pipeline completed in {(time.time() - t0) / 60:.1f} minutes")
    result = full_result.pipeline_result
    skeleton_path = Path(result.skeleton_path)
    weights_path = Path(result.skinning_weights_path)
    result_refined_diags = result.refinement_diagnostics

print("=== 5. diagnostics ===")
skeleton = json.loads(skeleton_path.read_text(encoding="utf-8"))
skinning = json.loads(weights_path.read_text(encoding="utf-8"))
print("articulation:", skeleton["articulation_type"])
print("joints:", [j["name"] for j in skeleton["joints"]])
print("bones :", [b["name"] for b in skeleton["bones"]])
print("root  :", skeleton["root_joint"])
print("skinned vertices:", len(skinning["weights"]))
print("max influences  :", skinning["max_influences"])

print("=== 6. preview renders ===")
try:
    import matplotlib
    import trimesh
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    refined = trimesh.load(str(REFINED_MESH_PATH), process=False)
    if isinstance(refined, trimesh.Scene):
        refined = trimesh.util.concatenate(tuple(refined.geometry.values()))

    normalized = refined.copy()
    scale = float(max(normalized.extents))
    normalized.apply_translation(-normalized.bounding_box.centroid)
    normalized.apply_scale(1.0 / scale)

    from contextlib import suppress as _suppress

    verts = normalized.vertices
    colors = None
    with _suppress(Exception):
        colors = normalized.visual.vertex_colors[:, :3] / 255.0

    joint_pos = {j["name"]: np.array(j["position"]) for j in skeleton["joints"]}
    bone_segments = [
        (joint_pos[b["parent_joint"]], joint_pos[b["child_joint"]]) for b in skeleton["bones"]
    ]
    joints_arr = np.array(list(joint_pos.values()))

    fig = plt.figure(figsize=(15, 5))
    views = [(10, 0, "front"), (10, 90, "side"), (80, 0, "top")]
    sample = np.random.default_rng(0).choice(len(verts), size=min(20000, len(verts)), replace=False)
    for i, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.scatter(
            verts[sample, 0],
            verts[sample, 2],
            verts[sample, 1],
            c=colors[sample] if colors is not None else "gray",
            s=1,
            alpha=0.6,
        )
        segs = [[(p[0], p[2], p[1]), (c[0], c[2], c[1])] for p, c in bone_segments]
        ax.add_collection3d(Line3DCollection(segs, colors="red", linewidths=2))
        ax.scatter(joints_arr[:, 0], joints_arr[:, 2], joints_arr[:, 1], c="red", s=30)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"rigged teddybear -- {title}")
        ax.set_box_aspect([1, 1, 1])
        ax.axis("off")
    plt.tight_layout()
    preview_path = OUT_DIR / "preview_rig.png"
    plt.savefig(preview_path, dpi=150)
    plt.close(fig)
    print("saved", preview_path)
except Exception as exc:  # noqa: BLE001 -- preview must never kill the run
    print("preview failed (non-fatal):", exc)

print("=== 7. package ===")
summary = {
    "articulation": skeleton["articulation_type"],
    "n_joints": len(skeleton["joints"]),
    "n_bones": len(skeleton["bones"]),
    "n_skinned_vertices": len(skinning["weights"]),
    "max_influences": skinning["max_influences"],
    "rigged_glb": str(RIGGED_GLB_PATH),
    "skeleton": str(skeleton_path),
    "weights": str(weights_path),
    "refined_mesh": str(REFINED_MESH_PATH),
    "refinement_diagnostics": result_refined_diags,
}
(OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))

print("\\nDONE. Outputs in", OUT_DIR)
