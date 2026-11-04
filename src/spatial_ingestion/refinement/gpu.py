"""Mesh-graph kernels for refinement, run on CUDA through torch.

Each function computes exactly what the VTK filter it replaces computes, so
moving a step onto the GPU changes its cost, not its output; tests/
test_refinement.py checks both against VTK. With no GPU the same torch code
runs on the CPU.

- `region_ids` replaces `split_bodies()`, which re-thresholds the whole mesh
  once per component: O(components x cells), and the refinement bottleneck on
  fragmented MASt3R output (bench A2). Labelling every cell in one pass and
  extracting the survivors once is O(cells).
- `edge_use_counts` replaces two `extract_feature_edges()` passes.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pyvista as pv

Device = Literal["auto", "cuda", "cpu"]


def resolve_device(requested: Device = "auto") -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' requested but torch.cuda.is_available() is False")
    return requested


def cell_arrays(mesh: pv.DataSet) -> tuple[np.ndarray, np.ndarray] | None:
    """(offsets, connectivity) of every cell, or None for meshes these kernels skip.

    PolyData that mixes verts, lines or strips in with its polygons is left to
    VTK, as are empty cells; neither occurs in reconstruction output.
    """
    if isinstance(mesh, pv.PolyData):
        if mesh.n_verts or mesh.n_lines or mesh.n_strips:
            return None
        cells = mesh.GetPolys()
    elif isinstance(mesh, pv.UnstructuredGrid):
        cells = mesh.GetCells()
    else:
        return None
    offsets = np.asarray(pv.convert_array(cells.GetOffsetsArray()), dtype=np.int64)
    connectivity = np.asarray(pv.convert_array(cells.GetConnectivityArray()), dtype=np.int64)
    if len(offsets) < 2 or np.any(np.diff(offsets) <= 0):
        return None
    return offsets, connectivity


def region_ids(
    n_points: int, offsets: np.ndarray, connectivity: np.ndarray, device: str
) -> np.ndarray:
    """Connected-component id of every cell, numbered as `split_bodies()` numbers them.

    Cells are connected when they share a point id. pyvista runs
    vtkConnectivityFilter with CELL_COUNT_DESCENDING: largest region first,
    and among equal sizes the region VTK seeded later (from a higher lowest
    cell id) first. Region k here is `split_bodies()[k]`.
    """
    import torch

    offsets_t = torch.as_tensor(offsets, device=device)
    conn = torch.as_tensor(connectivity, device=device)
    n_cells = len(offsets) - 1
    sizes = offsets_t[1:] - offsets_t[:-1]
    first_point = conn[offsets_t[:-1]]
    # Star edges: each cell's first point to each of its points.
    u = torch.repeat_interleave(first_point, sizes)
    v = conn

    # Union-find by hooking roots onto smaller roots, then pointer jumping.
    # Parents only ever decrease, so no cycles form.
    parent = torch.arange(n_points, device=device)
    while True:
        while True:
            jumped = parent[parent]
            if torch.equal(jumped, parent):
                break
            parent = jumped
        root_u, root_v = parent[u], parent[v]
        split = root_u != root_v
        if not bool(split.any()):
            break
        high = torch.maximum(root_u[split], root_v[split])
        low = torch.minimum(root_u[split], root_v[split])
        parent.scatter_reduce_(0, high, low, reduce="amin")

    cell_root = parent[first_point]
    _, inverse = torch.unique(cell_root, return_inverse=True)
    n_regions = int(inverse.max()) + 1
    first_cell = torch.full((n_regions,), n_cells, device=device, dtype=torch.int64)
    first_cell.scatter_reduce_(
        0, inverse, torch.arange(n_cells, device=device), reduce="amin", include_self=True
    )
    counts = torch.bincount(inverse, minlength=n_regions)
    # Sort key (count, first cell), both descending; first_cell < n_cells.
    order = torch.argsort(counts * n_cells + first_cell, descending=True)
    rank = torch.empty_like(first_cell)
    rank[order] = torch.arange(n_regions, device=device)
    return rank[inverse].cpu().numpy()


def edge_use_counts(
    n_points: int, offsets: np.ndarray, connectivity: np.ndarray, device: str
) -> dict[str, int]:
    """Boundary (used once) and non-manifold (used 3+ times) edge counts.

    Matches vtkFeatureEdges for triangle meshes, where "cells sharing both
    points" and "cells sharing the edge" coincide; callers pass only those.
    """
    import torch

    offsets_t = torch.as_tensor(offsets, device=device)
    conn = torch.as_tensor(connectivity, device=device)
    sizes = offsets_t[1:] - offsets_t[:-1]
    cell = torch.repeat_interleave(torch.arange(len(sizes), device=device), sizes)
    position = torch.arange(len(conn), device=device)
    # Each corner's edge runs to the next corner, the last wrapping to the first.
    following = torch.where(position + 1 == offsets_t[1:][cell], offsets_t[:-1][cell], position + 1)
    a, b = conn, conn[following]
    key = torch.minimum(a, b) * n_points + torch.maximum(a, b)
    _, uses = torch.unique(key, return_counts=True)
    return {
        "boundary_edge_count": int((uses == 1).sum()),
        "non_manifold_edge_count": int((uses > 2).sum()),
    }
