#!/bin/bash
set -e

echo ">>> 激活 conda 环境 polymetis ..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate polymetis || { echo "❌ 激活失败"; exit 1; }

POLY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)/polymetis"
export POLYMETIS_BUILD_DIR="$POLY_ROOT/polymetis/build_eef"
export PYTHONPATH="$POLYMETIS_BUILD_DIR:$POLY_ROOT/polymetis/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$POLYMETIS_BUILD_DIR/torch_isolation/pinocchio_isolation:$POLYMETIS_BUILD_DIR/torch_isolation${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

WORK_DIR="$POLY_ROOT/polymetis/python/polymetis"
if [[ ! -d "$WORK_DIR/conf" ]]; then
    echo "❌ 配置目录 $WORK_DIR/conf 不存在"
    exit 1
fi

echo ">>> 启动 Franka 客户端 ..."
cd "$WORK_DIR"
if [[ "${1:-}" == "--home" ]]; then
    shift
    echo ">>> 执行一次夹爪回零，夹爪手指将移动 ..."
    set -- "+gripper.executable_cfg.home_on_start=true" "$@"
fi
python ../scripts/launch_gripper.py gripper=franka_hand "$@"
