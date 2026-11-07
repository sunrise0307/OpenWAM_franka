#!/usr/bin/env bash
set -eo pipefail

# Edit this section once for each task and collection session.
TASK_ID=stack_bowl
TASK_PROMPT='Stack the two pink bowls in the center of the table.'
SUCCESS_CRITERION='释放后稳定停留至少一秒'
OUTPUT_DIR=/home/pnp/yzx/data
TRAIN_EPISODES=80
VAL_EPISODES=20
OPERATOR=operator01
SESSION_ID=sessionA
HEAD_SERIAL=263022075238
WRIST_SERIAL=262922070623
GELLO_PORT= # Leave empty to auto-detect the connected port in PORT_CONFIG_MAP.
CALIBRATION_DIR= # Optional directory of existing calibration files to copy per episode.
TELEOP_HZ=100

if [[ $# -ne 0 ]]; then
  printf 'Usage: bash %s\n' "$0" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /home/pnp/miniconda3/etc/profile.d/conda.sh
conda activate polymetis
# Conda activation hooks may read optional environment variables without defaults.
set -u
export POLYMETIS_BUILD_DIR=/home/pnp/yzx/frankateleop/polymetis/polymetis/build_eef
export PYTHONPATH="/home/pnp/yzx/.deps:$POLYMETIS_BUILD_DIR:/home/pnp/yzx/frankateleop/teleop:/home/pnp/yzx/frankateleop/polymetis/polymetis/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$POLYMETIS_BUILD_DIR/torch_isolation/pinocchio_isolation:$POLYMETIS_BUILD_DIR/torch_isolation${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ -z $GELLO_PORT ]]; then
  GELLO_PORT="$(python "$SCRIPT_DIR/detect_gello_port.py")"
  printf 'Detected GELLO port: %s\n' "$GELLO_PORT"
fi

calibration_args=()
if [[ -n $CALIBRATION_DIR ]]; then
  calibration_args=(--calibration-dir "$CALIBRATION_DIR")
fi

python "$SCRIPT_DIR/interactive_collector.py" \
  --output-dir "$OUTPUT_DIR" \
  --task-id "$TASK_ID" \
  --instruction "$TASK_PROMPT" --success-criterion "$SUCCESS_CRITERION" \
  --operator "$OPERATOR" --session-id "$SESSION_ID" \
  --head-serial "$HEAD_SERIAL" --wrist-serial "$WRIST_SERIAL" \
  --gello-port "$GELLO_PORT" "${calibration_args[@]}" \
  --train-episodes "$TRAIN_EPISODES" --val-episodes "$VAL_EPISODES" \
  --teleop-hz "$TELEOP_HZ"
