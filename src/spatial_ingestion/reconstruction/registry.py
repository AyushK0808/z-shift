from __future__ import annotations

from spatial_ingestion.reconstruction.backends.base import ReconstructionBackend
from spatial_ingestion.reconstruction.backends.mast3r import Mast3rBackend
from spatial_ingestion.reconstruction.models import ReconstructionJob


class ReconstructionBackendRegistry:
    def __init__(self, backends: list[ReconstructionBackend] | None = None) -> None:
        self._backends = {backend.name: backend for backend in backends or [Mast3rBackend()]}

    def get(self, name: str) -> ReconstructionBackend:
        try:
            return self._backends[name]
        except KeyError as exc:
            raise ValueError(f"Unknown reconstruction backend: {name}") from exc

    def resolve_for_job(self, job: ReconstructionJob) -> ReconstructionBackend:
        backend = self.get(job.backend_name)
        if not backend.supports(job):
            raise ValueError(
                f"Backend {backend.name} does not support reconstruction mode {job.mode}"
            )
        return backend
