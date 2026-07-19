"""Compatibility wrapper for the refinement pipeline API."""

from spatial_ingestion.refinement import MeshCleaningConfig, MeshProcessingError, MeshValidationError, clean_mesh

__all__ = [
    "MeshCleaningConfig",
    "MeshProcessingError",
    "MeshValidationError",
    "clean_mesh",
]