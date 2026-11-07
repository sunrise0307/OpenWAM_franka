#!/bin/bash
set -e

echo ">>> 激活 conda 环境 polymetis ..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate polymetis || { echo "❌ 激活失败，请确认 polymetis 环境存在"; exit 1; }

echo ">>> 定位 teleop ..."
POLY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)/teleop"
PROJECT_ROOT="$(dirname "$POLY_ROOT")"
export POLYMETIS_BUILD_DIR="$PROJECT_ROOT/polymetis/polymetis/build_eef"
export PYTHONPATH="$POLYMETIS_BUILD_DIR:$POLY_ROOT:$PROJECT_ROOT/polymetis/polymetis/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$POLYMETIS_BUILD_DIR/torch_isolation/pinocchio_isolation:$POLYMETIS_BUILD_DIR/torch_isolation${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ ! -d "$POLY_ROOT" ]]; then
    echo "❌ 未找到 teleop 文件夹：$POLY_ROOT"
    exit 1
fi

SCRIPT_PATH="$POLY_ROOT/experiments/run_env.py"
if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "❌ 未找到脚本：$SCRIPT_PATH"
    exit 1
fi

echo ">>> 启动Gripper 客户端 ..."
RUN_ARGS=(--agent teleop --use-save-interface --hz 100)
if [[ "${CAPTURE_RGB:-0}" == "1" ]]; then
    RUN_ARGS+=(--capture-rgb)
fi
python3 "$SCRIPT_PATH" "${RUN_ARGS[@]}"
#如果要启用采集数据，需要在后面增加“--use_save_interface”
