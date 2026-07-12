import sys
from pathlib import Path

from spatial_ingestion.reconstruction.runners.mast3r import configure_local_mast3r_imports


def test_configure_local_mast3r_imports_prefers_repo_local_checkout(tmp_path: Path) -> None:
    mast3r_root = tmp_path / "mast3r"
    dust3r_root = mast3r_root / "dust3r"
    dust3r_root.mkdir(parents=True)

    before = list(sys.path)
    try:
        configure_local_mast3r_imports(mast3r_root)
        prefixes = set(sys.path[:2])
        assert prefixes == {str(mast3r_root.resolve()), str(dust3r_root.resolve())}
    finally:
        sys.path[:] = before
