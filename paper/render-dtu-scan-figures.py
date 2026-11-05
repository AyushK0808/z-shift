"""Builds fig6-style panels for the five DTU scans, for choosing what to print.

Each scan's input photograph next to its reconstruction, rendered from the run's
own estimate of the camera that took the photograph. Unlike
render-case-study-mesh.py, the render uses that pose exactly and is not cropped,
so it lines up with the photograph. B6 saved only the raw mesh, so the refined meshes are
built here with the settings B10 scores. Writes into figures/:

    dtu-<scan>-input.png      the photograph
    dtu-<scan>-raw.png        the raw mesh
    dtu-<scan>-object.png     refined, object mode
    dtu-<scan>-isolated.png   refined, object mode with object isolation
    fig-dtu-<scan>.png        input | isolated, at \\columnwidth like fig6

    ZSHIFT_DTU_DIR=~/zshift/dtu ZSHIFT_B6_DIR=~/zshift/tier_b/full/outputs/b6 \\
        python paper/render-dtu-scan-figures.py [scan ...]

Requires pyvista with an off-screen backend, matplotlib and Pillow, plus the
project's refinement package. The camera index can be changed with
ZSHIFT_DTU_CAMERA (default 0, view 001 of the 10 spread views).
"""

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from PIL import Image

from spatial_ingestion.refinement import (
    MeshCleaningConfig,
    clean_mesh,
    load_mesh_file,
    to_trimesh,
)

SCANS = ("scan24", "scan37", "scan40", "scan55", "scan63")
SMOOTHING_ITERS = 15
WIDTH, HEIGHT = 1600, 1200
# MASt3R's focal length is in pixels of its 512-wide input.
INPUT_WIDTH = 512

os.chdir(os.path.dirname(os.path.abspath(__file__)))
dtu_dir = Path(os.environ.get("ZSHIFT_DTU_DIR", "~/zshift/dtu")).expanduser()
b6_dir = Path(os.environ.get("ZSHIFT_B6_DIR", "~/zshift/tier_b/full/outputs/b6")).expanduser()
camera_index = int(os.environ.get("ZSHIFT_DTU_CAMERA", "0"))


def render(mesh, camera):
    """Render `mesh` from `camera`'s estimated pose."""
    colors = next(
        (name for name in ("COLOR_0", "RGBA", "RGB", "rgb") if name in mesh.point_data), None
    )
    cam2world = np.asarray(camera["cam2world"], dtype=float)
    # OpenCV axes: +Y points down the image, +Z along the view.
    position = cam2world[:3, 3]
    forward = cam2world[:3, 2]
    down = cam2world[:3, 1]
    centre = np.asarray(mesh.center)
    focal_point = position + forward * np.linalg.norm(centre - position)

    input_height = INPUT_WIDTH * HEIGHT / WIDTH
    view_angle = 2 * np.degrees(np.arctan(0.5 * input_height / float(camera["focal_px"][0])))

    plotter = pv.Plotter(off_screen=True, window_size=(WIDTH, HEIGHT))
    plotter.set_background("white")
    plotter.add_mesh(
        mesh,
        scalars=colors,
        rgb=colors is not None,
        color=None if colors else "tan",
        smooth_shading=True,
        ambient=0.35,
        diffuse=0.7,
        specular=0.05,
    )
    plotter.camera.position = position
    plotter.camera.focal_point = focal_point
    plotter.camera.up = -down
    plotter.camera.view_angle = view_angle
    plotter.reset_camera_clipping_range()
    image = plotter.screenshot(None, return_img=True)
    plotter.close()

    return Image.fromarray(image[..., :3])


def compose(photo, render_img, n_views, target):
    """Input | render, with the layout and labels of fig6-case-study-validation.png."""
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Georgia", "Times New Roman", "DejaVu Serif"]
    fig_w = 3.5
    side_margin, top_margin, bottom_margin, gap, title_h = 0.04, 0.03, 0.04, 0.08, 0.17
    img_w = (fig_w - 2 * side_margin - gap) / 2
    height_a = img_w * photo.height / photo.width
    height_b = img_w * render_img.height / render_img.width
    row_h = max(height_a, height_b)
    fig_h = top_margin + row_h + title_h + bottom_margin
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=600)
    y_row = bottom_margin + title_h
    specs = [
        (side_margin, height_a, photo, f"(a) input capture (1 of {n_views} views)"),
        (side_margin + img_w + gap, height_b, render_img, "(b) reconstructed + refined mesh"),
    ]
    for x_in, h_in, img, title in specs:
        y_in = y_row + (row_h - h_in) / 2
        ax = fig.add_axes([x_in / fig_w, y_in / fig_h, img_w / fig_w, h_in / fig_h])
        ax.imshow(np.asarray(img))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#94a3b8")
            spine.set_linewidth(0.8)
        fig.text(
            (x_in + img_w / 2) / fig_w,
            (bottom_margin + title_h / 2) / fig_h,
            title,
            ha="center",
            va="center",
            fontsize=6,
            color="#0f172a",
        )
    fig.savefig(target, dpi=600, facecolor="white")
    plt.close(fig)


def refine(raw, isolate):
    config = MeshCleaningConfig(
        mode="object",
        smoothing_iters=SMOOTHING_ITERS,
        isolate_object=isolate,
        component_filter="labels",
    )
    return clean_mesh(raw, config)["mesh"]


def as_render_mesh(mesh):
    """Back through trimesh, so raw and refined carry colors the same way."""
    return pv.wrap(to_trimesh(mesh))


def main(scans):
    for scan in scans:
        run_dir = b6_dir / scan
        cameras = json.loads((run_dir / "cameras.json").read_text(encoding="utf-8"))["cameras"]
        camera = cameras[camera_index]
        name = os.path.basename(camera["image"])
        photo = Image.open(dtu_dir / "Rectified" / scan / name).convert("RGB")
        photo.save(f"figures/dtu-{scan}-input.png")

        raw = load_mesh_file(run_dir / f"{scan}.glb")
        print(scan, "point data:", list(raw.point_data), flush=True)
        variants = {
            "raw": raw,
            "object": refine(raw, False),
            "isolated": refine(raw, True),
        }
        renders = {}
        for label, mesh in variants.items():
            renders[label] = render(mesh, camera)
            renders[label].save(f"figures/dtu-{scan}-{label}.png")
        compose(photo, renders["isolated"], len(cameras), f"figures/fig-dtu-{scan}.png")
        print(f"{scan}: wrote input, raw, object, isolated, composite (camera {name})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or SCANS)
