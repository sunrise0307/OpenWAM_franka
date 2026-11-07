#!/bin/bash
set -e

echo ">>> 激活 conda 环境 polymetis ..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate polymetis || { echo "❌ 激活失败"; exit 1; }

echo ">>> 清理旧 run_server 进程 ..."
sudo pkill -9 run_server || echo "⚠️ 未发现 run_server 进程或无需清理"

POLY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)/polymetis"
export POLYMETIS_BUILD_DIR="$POLY_ROOT/polymetis/build_eef"

WORK_DIR="$POLY_ROOT/polymetis/python/polymetis"
export PYTHONPATH="$POLYMETIS_BUILD_DIR:$POLY_ROOT/polymetis/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$POLYMETIS_BUILD_DIR/torch_isolation/pinocchio_isolation:$POLYMETIS_BUILD_DIR/torch_isolation${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ ! -d "$WORK_DIR/conf" ]]; then
    echo "❌ 配置目录 $WORK_DIR/conf 不存在"
    exit 1
fi

echo ">>> 启动Franka 客户端 ..."
cd "$WORK_DIR"
python ../scripts/launch_robot.py robot_model=fr3 robot_client=franka_hardware
