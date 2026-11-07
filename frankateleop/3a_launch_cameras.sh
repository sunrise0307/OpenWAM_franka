#!/bin/bash
set -e

: "${WRIST_CAMERA_SERIAL:?Set WRIST_CAMERA_SERIAL}"
: "${BASE_CAMERA_SERIAL:?Set BASE_CAMERA_SERIAL}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate polymetis
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)"
export PYTHONPATH="$PROJECT_ROOT/teleop${PYTHONPATH:+:$PYTHONPATH}"
python3 "$PROJECT_ROOT/teleop/experiments/launch_camera_nodes.py" \
    --color-only --wrist-serial "$WRIST_CAMERA_SERIAL" \
    --base-serial "$BASE_CAMERA_SERIAL"
