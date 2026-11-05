"""Reconstruction backends: what the stage's replaceability rests on.

Phase 2 reads images and writes three artifacts, a mesh, `point_cloud.ply` and
`cameras.json`, plus a run manifest (`export.write_scene_artifacts`). A backend
is whatever turns images into a scene and a scene into those artifacts; refinement,
routing and rigging see only the artifacts, so swapping the backend changes
nothing after it. `Mast3rBackend` is the default and `Dust3rBackend` is a second,
independent network and optimiser behind the same two calls.

A backend is selected by `ReconstructionJob.backend_name`. Register another
with `register_backend`; the contract test (tests/test_backends.py) states what
it has to produce.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from spatial_ingestion.reconstruction.alignment import run_sparse_alignment, select_pairs
from spatial_ingestion.reconstruction.config import DEFAULT_MODEL_NAME
from spatial_ingestion.reconstruction.export import export_scene_to_mesh, write_scene_artifacts
from spatial_ingestion.reconstruction.inference import load_images
from spatial_ingestion.reconstruction.models import HandoffFrame, SyncViewGroup

logger = logging.getLogger(__name__)

DUST3R_MODEL_NAME = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
# Global alignment iterations for DUSt3R's point-cloud optimiser, the values its
# own demo uses.
_DUST3R_ALIGNMENT_ITERATIONS = 300
_DUST3R_ALIGNMENT_LR = 0.01

_dust3r_models: dict[str, object] = {}


class ReconstructionBackend(ABC):
    """Images in, scene out; scene in, the stage's artifacts out."""

    name: str
    default_model: str

    def resolve_model(self, requested: str) -> str:
        """The checkpoint to load: the request, unless it is another backend's default."""
        return self.default_model if requested in _ALL_DEFAULT_MODELS else requested

    @abstractmethod
    def align(
        self,
        *,
        image_paths: list[Path],
        output_dir: Path,
        model_name: str,
        device: str,
        image_size: int,
        pairing_strategy: str,
        sync_view_groups: list[SyncViewGroup] | None,
        frames: list[HandoffFrame] | None,
        stats: dict[str, Any],
    ) -> object:
        """Estimate cameras and per-view geometry for `image_paths`."""

    @abstractmethod
    def export(
        self,
        scene: object,
        output_path: Path,
        output_dir: Path,
        *,
        tsdf_thresh: float,
        min_conf_thr: float,
        stats: dict[str, Any],
    ) -> bool:
        """Write the mesh, point cloud and cameras; return whether fusion fell back."""


class Mast3rBackend(ReconstructionBackend):
    name = "mast3r"
    default_model = DEFAULT_MODEL_NAME

    def align(self, **kwargs: Any) -> object:
        return run_sparse_alignment(**kwargs)

    def export(
        self,
        scene: object,
        output_path: Path,
        output_dir: Path,
        *,
        tsdf_thresh: float,
        min_conf_thr: float,
        stats: dict[str, Any],
    ) -> bool:
        return export_scene_to_mesh(
            scene,
            output_path,
            output_dir,
            tsdf_thresh=tsdf_thresh,
            min_conf_thr=min_conf_thr,
            stats=stats,
        )


class Dust3rBackend(ReconstructionBackend):
    """DUSt3R's pairwise network and its dense global point-cloud optimiser.

    The same pair graph as MASt3R, but a different network (no matching head),
    a different optimiser (`PointCloudOptimizer` over dense pointmaps, not
    sparse-matching alignment) and no TSDF fusion: the pointmaps are masked by
    confidence and meshed per view. Its output is not metric, so the DTU camera
    fit supplies the scale.
    """

    name = "dust3r"
    default_model = DUST3R_MODEL_NAME

    def align(
        self,
        *,
        image_paths: list[Path],
        output_dir: Path,
        model_name: str,
        device: str,
        image_size: int,
        pairing_strategy: str,
        sync_view_groups: list[SyncViewGroup] | None,
        frames: list[HandoffFrame] | None,
        stats: dict[str, Any],
    ) -> object:
        try:
            from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
            from dust3r.inference import inference
            from dust3r.model import AsymmetricCroCo3DStereo
        except ImportError as exc:
            raise RuntimeError(
                "DUSt3R is not installed. Run scripts/setup-mast3r.sh, which installs "
                "third_party/mast3r/dust3r"
            ) from exc

        model = _dust3r_models.get(model_name)
        if model is None:
            model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
            model.eval()  # type: ignore[attr-defined]
            _dust3r_models[model_name] = model

        images = load_images(image_paths, image_size=image_size)
        pairs = select_pairs(image_paths, images, pairing_strategy, sync_view_groups, stats=stats)
        output = inference(pairs, model, device, batch_size=1, verbose=False)
        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
        scene.compute_global_alignment(
            init="mst",
            niter=_DUST3R_ALIGNMENT_ITERATIONS,
            schedule="cosine",
            lr=_DUST3R_ALIGNMENT_LR,
        )
        # MASt3R's scene carries this natively; the optimiser orders its views by
        # input index, so the paths line up with it view for view.
        scene.img_paths = [str(path) for path in image_paths]
        return scene

    def export(
        self,
        scene: object,
        output_path: Path,
        output_dir: Path,
        *,
        tsdf_thresh: float,
        min_conf_thr: float,
        stats: dict[str, Any],
    ) -> bool:
        from dust3r.utils.device import to_numpy

        if tsdf_thresh > 0:
            logger.info("DUSt3R has no TSDF fusion; ignoring tsdf_thresh=%s", tsdf_thresh)
        imgs = to_numpy(scene.imgs)  # type: ignore[attr-defined]
        pts3d = to_numpy(scene.get_pts3d())  # type: ignore[attr-defined]
        confs = [to_numpy(conf) for conf in scene.im_conf]  # type: ignore[attr-defined]
        write_scene_artifacts(
            imgs,
            pts3d,
            confs,
            scene,
            output_path,
            output_dir,
            min_conf_thr=min_conf_thr,
            tsdf_fell_back=False,
            stats=stats,
        )
        stats["fusion"] = "none"
        return False


_BACKENDS: dict[str, ReconstructionBackend] = {}
_ALL_DEFAULT_MODELS: set[str] = set()


def register_backend(backend: ReconstructionBackend) -> None:
    _BACKENDS[backend.name] = backend
    _ALL_DEFAULT_MODELS.add(backend.default_model)


def get_backend(name: str) -> ReconstructionBackend:
    try:
        return _BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"Unknown reconstruction backend '{name}'; available: {sorted(_BACKENDS)}"
        ) from None


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


register_backend(Mast3rBackend())
register_backend(Dust3rBackend())
