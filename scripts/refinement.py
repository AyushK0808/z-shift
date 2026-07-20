"""Compatibility wrapper and CLI entry point for the refinement pipeline API."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial_ingestion.refinement import (  # noqa: E402
    MeshCleaningConfig,
    MeshProcessingError,
    MeshValidationError,
    clean_ai_mesh,
    clean_mesh,
)
from spatial_ingestion.refinement.cli import main  # noqa: E402

__all__ = [
    "MeshCleaningConfig",
    "MeshProcessingError",
    "MeshValidationError",
    "clean_ai_mesh",
    "clean_mesh",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())