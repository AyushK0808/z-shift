#!/usr/bin/env bash
set -euo pipefail

REPO="https://github.com/naver/mast3r"
PINNED="f5209afc300cec36239a7ac992263f36847bbba0"
TARGET="$(cd "$(dirname "$0")/.." && pwd)/third_party/mast3r"

if [ -d "$TARGET" ]; then
  echo "mast3r already exists at $TARGET"
  exit 0
fi

echo "Cloning mast3r into $TARGET ..."
git clone "$REPO" "$TARGET"
git -C "$TARGET" checkout "$PINNED"
git -C "$TARGET" submodule update --init --recursive
echo "Done."
