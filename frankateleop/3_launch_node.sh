#!/bin/bash
set -e

echo ">>> 激活 conda 环境 polymetis ..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate polymetis || { echo "❌ 激活失败，请确认 polymetis 环境存在"; exit 1; }

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)"
export POLYMETIS_BUILD_DIR="$PROJECT_ROOT/polymetis/polymetis/build_eef"
export PYTHONPATH="$PROJECT_ROOT/teleop:$PROJECT_ROOT/polymetis/polymetis/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$POLYMETIS_BUILD_DIR/torch_isolation/pinocchio_isolation:$POLYMETIS_BUILD_DIR/torch_isolation${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo ">>> 在当前目录及子目录中查找 teleop ..."
POLY_ROOT=$(find . -type d -name "teleop" | head -n 1)

if [[ -z "$POLY_ROOT" ]]; then
    echo "❌ 当前目录下未找到 teleop 文件夹"
    exit 1
fi

SCRIPT_PATH="$POLY_ROOT/experiments/launch_nodes.py"
if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "❌ 未找到脚本：$SCRIPT_PATH"
    exit 1
fi

echo ">>> 启动Robot Node ..."
python3 "$SCRIPT_PATH" --robot=fr3
