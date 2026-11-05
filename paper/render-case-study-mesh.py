"""Renders the case study's refined mesh from its first input camera.

A scripted stand-in for the hand-made Blender render (outputs/blender-output.png)
that panel (b) of fig6 used: the camera is the reconstruction's own estimate
for the first input frame, which is the photograph in panel (a), so the two
panels show the subject from the same side.

    ZSHIFT_CASE_STUDY_DIR=~/zshift/case_study_24/run \\
        python paper/render-case-study-mesh.py outputs/case-study-render.png

Requires pyvista with an off-screen backend (VTK's EGL/OSMesa build works
headless under WSL2). Reads <run>/mesh_refined.glb and <run>/cameras.json.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

os.chdir(os.path.dirname(os.path.abspath(__file__)))
run_dir = Path(os.environ.get("ZSHIFT_CASE_STUDY_DIR", "")).expanduser()
if not (run_dir / "mesh_refined.glb").exists():
    raise SystemExit(f"set ZSHIFT_CASE_STUDY_DIR to a case-study run dir; got {run_dir!s}")
target = sys.argv[1] if len(sys.argv) > 1 else "outputs/case-study-render.png"

mesh = pv.read(run_dir / "mesh_refined.glb")
if isinstance(mesh, pv.MultiBlock):
    mesh = mesh.combine().extract_surface(algorithm=None)
colors = next((name for name in ("COLOR_0", "RGBA", "RGB", "rgb") if name in mesh.point_data), None)

camera = json.loads((run_dir / "cameras.json").read_text(encoding="utf-8"))["cameras"][0]
cam2world = np.asarray(camera["cam2world"], dtype=float)
# OpenCV axes: +Y points down the image.
down = cam2world[:3, 1]
centre = np.asarray(mesh.center)
offset = centre - cam2world[:3, 3]
# Back off a little along the camera's line to the subject, so it all fits.
position = centre - offset * 1.15

WIDTH, HEIGHT = 1600, 900
# MASt3R's focal length is in pixels of its 512-wide input.
input_height = 512 * HEIGHT / WIDTH
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
plotter.camera.focal_point = centre
plotter.camera.up = -down
plotter.camera.view_angle = view_angle
plotter.reset_camera_clipping_range()
image = plotter.screenshot(None, return_img=True)

# Crop to the subject, then pad back to the input photograph's aspect so the
# two panels of fig6 print at the same height.
from PIL import Image  # noqa: E402

photo = Image.open("outputs/teddy.jpeg")
aspect = photo.width / photo.height
rows, cols = np.nonzero(np.any(image[..., :3] < 245, axis=2))
pad = int(0.08 * max(rows.ptp(), cols.ptp()))
top, bottom = max(rows.min() - pad, 0), min(rows.max() + pad, image.shape[0])
left, right = max(cols.min() - pad, 0), min(cols.max() + pad, image.shape[1])
crop = Image.fromarray(image[top:bottom, left:right, :3])
width = max(crop.width, round(crop.height * aspect))
height = max(crop.height, round(width / aspect))
canvas = Image.new("RGB", (width, height), "white")
canvas.paste(crop, ((width - crop.width) // 2, (height - crop.height) // 2))
canvas.save(target)
print(f"wrote {target} from {run_dir} (camera: {camera['image']})")
