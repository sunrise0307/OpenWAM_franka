#!/bin/bash
set -e

echo ">>> 激活 conda 环境 polymetis ..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate polymetis || { echo "❌ 激活失败，请确认 polymetis 环境存在"; exit 1; }

echo ">>> 在当前目录及子目录中查找 teleop ..."
POLY_ROOT=$(find ../ -type d -name "teleop" | head -n 1)

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
python3 "$SCRIPT_PATH" --robot=fr3_left --tele_port=6002 --robot_port=50052 --gripper_port=50054 --robot_ip=127.0.0.1
#robot_ip是直接连接机器人的主机的IP，如果是本机直连机器人，可以用127.0.0.1代替
