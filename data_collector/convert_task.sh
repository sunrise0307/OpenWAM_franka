#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /home/pnp/miniconda3/etc/profile.d/conda.sh
conda activate polymetis
export PYTHONPATH="/home/pnp/yzx/.deps${PYTHONPATH:+:$PYTHONPATH}"
python "$SCRIPT_DIR/convert_lerobot_v3.py" "$@"
