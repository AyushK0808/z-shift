# Auto-rigging: plan to make Phase 5 produce usable rigs

Companion to `AUTO_RIGGING.md` (the research survey). This is the
implementation plan for the rigger in `src/spatial_ingestion/auto_rigging/`,
driven by what it does today on a real capture.

## What is wrong today (measured)

On the 24-view CO3D teddybear rig with the floor removed
(`~/zshift/case_study_24/run/rig`), in units where the mesh's longest side is 1:

| Check | Result |
|---|---|
| Up axis | Template assumes +Y; the floor's normal is 48.5 degrees from it |
| Joint placement | Plotted against the mesh, at least `head`, `left_hand`, `left_foot` and `right_foot` fall outside its silhouette (judged by eye; the open mesh has no inside to test numerically yet) |
| Weight coverage | `right_hand` dominates 36% of vertices and `right_foot` 25%; `left_hand` 1.2%, `left_foot` 0.7% |
| Weight sharpness | 98.7% of vertices have no weight above 0.5 |

Without `isolate_object` the rig is fitted mostly or entirely to the floor
(FINDINGS #12).

fig13's rigging panel gives 1.0 to all of its contract checks (weights sum to
1, at most 4 influences, every vertex bound), none of which ask whether the rig
is right. Its one zero, "no warnings", is a warning the code emits on every run.

Root causes, in `skeleton/templates.py` and `skinning/inverse_distance.py`:

1. **No orientation.** Joints are fixed fractions of the axis-aligned bounding
   box, with Y as height. MASt3R's world frame has no relation to gravity, so
   the template is applied along an arbitrary axis.
2. **No shape.** A bounding box knows nothing about where the limbs are; the
   joints land in empty space. The A8 benchmark measured it: a 30 degree
   tilt moves the worst joint by 39% of the mesh's extent.
3. **Seven joints, no knees, elbows or neck.** Even placed perfectly, the
   template cannot bend a limb.
4. **Euclidean inverse distance to joint points.** Weights leak across gaps
   (an arm takes weight from the torso it is next to), never reach zero, and
   ignore bones (a vertex midway along a bone is bound to its two ends).
5. **The input is not riggable as-is.** The refined mesh is open,
   non-manifold (141K non-manifold edges at 24 views) and in thousands of
   fragments, so any volume- or surface-based method has to run on a proxy.

## Target and acceptance criteria

"Works" means, on every shape in the test set below, all of:

- **Orientation:** recovered up axis within 10 degrees of ground truth.
- **Joints inside:** every joint inside the proxy volume (winding number
  > 0.5); every bone at least 90% inside.
- **Joint accuracy** (synthetic and rigged-asset sets with a reference
  skeleton): CD-J2J below 5% of the bounding-box diagonal (RigNet's metric).
- **Skinning:** each joint dominates at least 1% of vertices; at least 80% of
  vertices have a dominant weight above 0.5; RigNet precision/recall above
  0.8 against reference weights.
- **Deformation:** rotating each limb 45 degrees under linear blend skinning
  flips under 0.5% of faces and changes volume by under 5%.
- **Honest failure:** if a gate fails, the rigger returns a single-root static
  rig and a machine-readable reason, not a broken biped. The warning becomes
  conditional on what was actually detected.

Nothing rigs every noisy, partial MASt3R mesh perfectly; the gates above make
"usable" testable and make the failures visible instead of silent.

## Plan

### Phase 0: measure before changing anything (1-2 days)

- Add the checks above to `bench/exp_a8_rigging_quality.py` as columns
  (`joint_inside_frac`, `bone_inside_frac`, `dominant_joint_coverage_min`,
  `sharp_vertex_frac`, `deform_volume_change`, `up_axis_error_deg`).
- Test set: A8's procedural capsule bipeds and quadrupeds, each under random
  SO(3) rotations (the template's blind spot); 20-50 rigged characters from
  Objaverse-XL (Apache-2.0) or Articulation-XL2.0 for joint and weight
  ground truth; the B9 teddybear for a real, noisy case.
- Replace fig13's contract-only rigging panel with these numbers, so the
  paper stops reporting 1.0 for a broken rig.

### Phase 1: canonical frame (2 days)

New `auto_rigging/orientation.py`:

- **Up:** from the reconstruction's own cameras (`cameras.json`, already
  written by Phase 2): the mean of each camera's image-up vector (-Y in OpenCV
  axes) for an upright capture. Cross-check against the support-plane normal
  that `isolate_object` already finds (`support_plane_normal` in the
  refinement diagnostics). Disagreement over 20 degrees lowers confidence.
- **Forward:** the horizontal direction most cameras looked from (subjects are
  photographed from the front), falling back to the horizontal PCA axis.
- **Lateral:** up x forward. Symmetry plane: refine with a reflection-ICP of
  the proxy about the up-forward plane.
- Rig in that frame, store the transform in `rigging_manifest.json`, and
  apply its inverse to the exported skeleton so the GLB stays in the mesh's
  frame. Pass the cameras and plane through `final_pipeline.handoff` (one new
  optional argument).

### Phase 2: a riggable proxy (2-3 days)

New `auto_rigging/proxy.py`:

- Input: the isolated object (`isolate_object=True`, now required for rigging).
- Closed, manifold proxy: voxelize the refined points at ~1/128 of the
  bounding-box diagonal, fill the interior with a flood fill from outside,
  close small gaps with a 2-voxel morphological close, then marching cubes.
  This handles open, non-manifold, fragmented input, which mesh repair
  (pymeshfix) does not. Decimate to 10-20K faces.
- All skeleton and skinning work runs on the proxy; weights are transferred
  back to the full mesh by barycentric interpolation at each vertex's closest
  proxy point.

### Phase 3: shape-aware skeleton (about a week)

Replace `TemplateSkeletonFitter` with an embedding behind the same `fit()`:

1. **Curve skeleton** of the proxy: 3D thinning of the voxel grid
   (`skimage.morphology.skeletonize`, BSD), converted to a graph; prune spurs
   shorter than 5% of the height; collapse junction clusters.
2. **Extremities:** graph leaves, labelled in the canonical frame: head =
   highest leaf above the torso, feet = the two lowest, hands = the two most
   lateral leaves above the hips, tail = a leaf behind the hips (quadrupeds).
3. **Embedding** (Pinocchio, Baran and Popovic 2007): place the richer
   template (add neck, two elbows, two knees, and shoulders and hips for the
   biped) on skeleton-graph nodes by minimising a cost of bone-length
   proportions, symmetry, staying on the medial axis, and extremity labels.
   It is a small discrete problem, solvable with A* or DP over graph paths.
4. **Fallbacks:** a limb the graph does not show (fused arms, occlusion) is
   placed from the template's proportions relative to found joints and
   flagged; fewer than three extremities means non-articulated, so a static rig.
5. Optional learned backend behind the same interface for later: UniRig or
   MagicArticulate (check licences; RigAnything is non-commercial, RigNet GPL).

### Phase 4: skinning that respects the shape (3 days)

Replace `InverseDistanceSkinner` with bone heat diffusion (Pinocchio's
"bone heat", the method Blender's automatic weights use):

- For each bone, solve `(L + H) w = H p` on the proxy, where `L` is the
  cotangent Laplacian, `H` a diagonal of 1/d^2 to the nearest *visible* bone
  segment (distance to the segment, not to its end joints; visibility by a ray
  test inside the proxy), and `p` is 1 where this bone is nearest.
- Laplacian from `robust_laplacian` (MIT; Sharp and Crane 2020), which
  tolerates the geometry that remains; solve with `scipy.sparse.linalg`.
- Keep the top 4, renormalise, transfer to the full mesh. Bounded biharmonic
  weights (libigl, MPL-2.0) are the heavier alternative if heat weights leak
  in tests.

### Phase 5: gates, tests, wiring (2 days)

- Run the acceptance checks inside `AutoRiggingPipeline.rig_mesh`; write them
  to `rigging_manifest.json`; on failure fall back to the static rig with the
  reason.
- Unit tests: rotated procedural bipeds recover up within 10 degrees and put
  every joint inside; a T-shape gets elbows at the bends; heat weights give a
  torso vertex zero arm weight; an open, fragmented sphere still produces a
  closed proxy.
- Re-run A8 and B9, and regenerate fig13 with the new rigging panel.

## Order and effort

Phase 0, then 1 and 2 together (orientation needs the proxy's extremities to
pick forward), then 3, then 4, then 5. About three weeks for one person.
Phases 0-2 alone fix the two worst defects (wrong axis, joints outside the
body) in under a week, and are worth landing first.
