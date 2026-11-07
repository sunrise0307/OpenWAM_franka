#!/bin/bash
set -e

echo ">>> 激活 conda 环境 polymetis ..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate polymetis || { echo "❌ 激活失败"; exit 1; }

POLY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)/polymetis"
export PYTHONPATH="$POLY_ROOT/polymetis/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$POLY_ROOT/polymetis/build/torch_isolation/pinocchio_isolation:$POLY_ROOT/polymetis/build/torch_isolation${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

WORK_DIR="$POLY_ROOT/polymetis/python/polymetis"
if [[ ! -d "$WORK_DIR/conf" ]]; then
    echo "❌ 配置目录 $WORK_DIR/conf 不存在"
    exit 1
fi

echo ">>> 启动 Franka 客户端 ..."
cd "$WORK_DIR"
python ../scripts/launch_gripper.py --config-name=launch_left_gripper # gripper=franka_hand  #1. franka_hand 2.none
