#!/bin/bash
set -e

echo ">>> 激活 conda 环境 polymetis ..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate polymetis || { echo "❌ 激活失败"; exit 1; }
echo "✅ conda环境激活成功"

echo ">>> 清理占用50051端口的进程 ..."

# 检查lsof是否安装
if ! command -v lsof &> /dev/null; then
    echo "❌ 错误：未找到lsof命令，请先安装（例如：sudo apt install lsof）"
    exit 1
fi

echo "🔍 正在查找占用50051端口的进程..."
PID=$(sudo lsof -t -i:50051 || true)
echo "📝 调试信息：查找到的PID=$PID"

if [[ -n "$PID" ]]; then
    echo "发现占用50051端口的进程，PID: $PID，正在终止..."
    if sudo kill -9 "$PID"; then
        echo "✅ 进程$PID已成功终止"
    else
        echo "❌ 终止进程$PID失败"
        exit 1
    fi
else
    echo "⚠️ 未发现占用50051端口的进程，无需清理"
fi

echo "✅ 端口清理步骤完成"

POLY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)/polymetis"
export PYTHONPATH="$POLY_ROOT/polymetis/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$POLY_ROOT/polymetis/build/torch_isolation/pinocchio_isolation:$POLY_ROOT/polymetis/build/torch_isolation${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "📝 调试信息：POLY_ROOT=$POLY_ROOT"

WORK_DIR="$POLY_ROOT/polymetis/python/polymetis"
echo "📝 调试信息：WORK_DIR=$WORK_DIR"

if [[ ! -d "$WORK_DIR/conf" ]]; then
    echo "❌ 配置目录 $WORK_DIR/conf 不存在"
    exit 1
fi
echo "✅ 配置目录检查通过"

echo ">>> 启动Franka 客户端 ..."
cd "$WORK_DIR" || { echo "❌ 无法进入工作目录 $WORK_DIR"; exit 1; }
echo "📝 调试信息：当前工作目录=$(pwd)"

python ../scripts/launch_robot.py --config-name=launch_right_robot

