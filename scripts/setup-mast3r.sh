#!/usr/bin/env bash
set -euo pipefail

REPO="https://github.com/naver/mast3r"
PINNED="f5209afc300cec36239a7ac992263f36847bbba0"
TARGET="$(cd "$(dirname "$0")/.." && pwd)/third_party/mast3r"

if [ -d "$TARGET" ]; then
  echo "mast3r already cloned at $TARGET"
else
  echo "Cloning mast3r into $TARGET ..."
  git clone "$REPO" "$TARGET"
  git -C "$TARGET" checkout "$PINNED"
  git -C "$TARGET" submodule update --init --recursive
fi

# Both repos are flat package dirs with no setup.py/pyproject.toml.
# Write minimal ones so pip install -e works without sys.path hacks.
# (third_party/ is gitignored so these won't pollute the repo.)

if [ ! -f "$TARGET/pyproject.toml" ]; then
  cat > "$TARGET/pyproject.toml" << 'PYEOF'
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "mast3r"
version = "0.1.0"
description = "MASt3R: Matching And Stereo 3D Reconstruction"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["."]
include = ["mast3r*"]
PYEOF
fi

DUST3R="$TARGET/dust3r"
if [ ! -f "$DUST3R/pyproject.toml" ]; then
  cat > "$DUST3R/pyproject.toml" << 'PYEOF'
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "dust3r"
version = "0.1.0"
description = "DUSt3R: Geometric 3D Vision Made Easy"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["."]
include = ["dust3r*"]
PYEOF
fi

echo "Installing mast3r Python package (editable) ..."
uv pip install -e "$TARGET"

echo "Installing dust3r Python package (editable) ..."
uv pip install -e "$DUST3R"

echo "Installing dependencies ..."
uv pip install -r "$TARGET/requirements.txt" 2>/dev/null || true
uv pip install -r "$DUST3R/requirements.txt" 2>/dev/null || true

echo "Compiling RoPE CUDA kernels (optional, speeds up positional embeddings) ..."
cd "$DUST3R/croco/models/curope"
if python setup.py build_ext --inplace 2>/dev/null; then
  echo "RoPE kernels compiled."
else
  echo "RoPE kernel compilation failed (non-CUDA env). PyTorch fallback will be used."
fi
cd - > /dev/null

echo ""
echo "mast3r setup complete."
echo "Verify: python -c 'from mast3r.model import AsymmetricMASt3R; print(\"OK\")'"
