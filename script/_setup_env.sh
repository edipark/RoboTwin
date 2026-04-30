#!/bin/bash
# Run this script from the repository root (third_party/RoboTwin/).
# Prerequisites: conda env "RoboTwin" must already exist with all packages installed.
#
# Usage:
#   cd /lustre/meat124/Soft-VLA/third_party/RoboTwin
#   bash script/_setup_env.sh [ASSETS_SOURCE_DIR]
#
# ASSETS_SOURCE_DIR defaults to /lustre/meat124/RoboTwin

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ASSETS_SRC="${1:-/lustre/meat124/RoboTwin}"

echo "=== RoboTwin env setup for: $REPO_ROOT ==="
echo "=== Using assets from: $ASSETS_SRC ==="

# 1. Symlink assets (embodiments, objects, background_texture)
ASSETS_DIR="$REPO_ROOT/assets"
for folder in embodiments objects background_texture; do
    SRC="$ASSETS_SRC/assets/$folder"
    DEST="$ASSETS_DIR/$folder"
    if [ -e "$DEST" ] || [ -L "$DEST" ]; then
        echo "[skip] $folder already exists at $DEST"
    else
        if [ ! -d "$SRC" ]; then
            echo "[error] Source not found: $SRC"
            echo "  -> Download assets first: bash script/_download_assets.sh in $ASSETS_SRC"
            exit 1
        fi
        ln -s "$SRC" "$DEST"
        echo "[done] Symlinked $folder -> $SRC"
    fi
done

# 2. Generate curobo.yml from *_tmp.yml templates (sets absolute paths)
echo "=== Updating embodiment config paths ==="
cd "$REPO_ROOT"
conda run -n RoboTwin python script/update_embodiment_config_path.py

echo ""
echo "=== Setup complete! ==="
echo "Run collect_data.sh on a GPU node:"
echo "  bash collect_data.sh <task_name> <task_config> <gpu_id>"
