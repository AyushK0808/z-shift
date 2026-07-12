from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
THIRD_PARTY_ROOT = BASE_DIR / "third_party"
MAST3R_ROOT = THIRD_PARTY_ROOT / "mast3r"
LOCAL_STORAGE_ROOT = BASE_DIR / "data" / "object_store"
NORMALIZED_OUTPUT_ROOT = BASE_DIR / "data" / "normalized"
RECONSTRUCTION_OUTPUT_ROOT = BASE_DIR / "data" / "reconstruction"
CHECKPOINT_ROOT = BASE_DIR / "data" / "checkpoints"
DEFAULT_IMAGE_SIZE = (1024, 1024)
LIVE_BUFFER_SIZE = 64
DEFAULT_MULTI_VIEW_BACKEND = "mast3r"
