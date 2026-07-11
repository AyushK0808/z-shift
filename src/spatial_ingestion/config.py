from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
LOCAL_STORAGE_ROOT = BASE_DIR / "data" / "object_store"
NORMALIZED_OUTPUT_ROOT = BASE_DIR / "data" / "normalized"
DEFAULT_IMAGE_SIZE = (1024, 1024)
LIVE_BUFFER_SIZE = 64

