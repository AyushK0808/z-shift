"""Guards and setup hints for the optional MASt3R/DUSt3R dependencies."""

from __future__ import annotations

MAST3R_SETUP_HINT = (
    "Run scripts/setup-mast3r.sh or pip install -e third_party/mast3r "
    "&& pip install -e third_party/mast3r/dust3r"
)


def mast3r_dependency_error(detail: str) -> RuntimeError:
    """Build the canonical 'MASt3R is not installed' error."""
    return RuntimeError(f"{detail} is not installed. {MAST3R_SETUP_HINT}")
