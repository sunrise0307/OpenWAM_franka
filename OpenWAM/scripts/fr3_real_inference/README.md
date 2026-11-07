# FR3 真机 policy 服务

本机运行 `policy_server.py`，机器人端作为 WebSocket client 发送观测、接收动作。请求沿用 `scripts/inference_test/inference_continuous_test.py` 的协议，**每条响应默认返回 8 个动作，`action` 形状为 `[8,10]`**（FR3 协议版本 2）。图像处理、归一化和位姿约定已按 `openwam/dataloader/fr3.py`、`configs/dataloader/fr3_stack_bowl.yaml` 及数据集 `meta/conversion.json` 对齐。

## 启动

在 OpenWAM 根目录执行，使用已安装项目依赖的 `openwam` 环境：

```bash
conda activate openwam
cd /home/aiworldlab/yzx/OpenWAM

# 只检查配置、统计量和文件是否齐全；不加载模型，不启动服务
python scripts/fr3_real_inference/policy_server.py --check-only

# 正式启动；默认使用数据集中的叠碗任务指令
python scripts/fr3_real_inference/policy_server.py \
  --host 0.0.0.0 --port 8848 --device cuda:0 \
  --denoise-steps 10 --inference-horizon 8
```

默认读取 `assets/fr3_stack_bowl_ckpt` 中 step 最大的权重，当前为 `checkpoint_step_4500.safetensors`。也可显式添加：

```bash
--ckpt-dir assets/fr3_stack_bowl_ckpt --ckpt-name checkpoint_step_4500.safetensors
```

默认任务文本取自数据集 `meta/tasks.parquet`：`Stack the two pink bowls in the center of the table.`。添加 `--prompt '任务文本'` 可覆盖默认值；请求中的非空 `prompt` 优先，文本原样传给模型。生产服务只需 checkpoint 和代码，启动不读取完整数据集。无需修改权重配置中的 `/path/to/Wan2.2-TI2V-5B`：此目录包含组件定义、完整权重和 tokenizer，使用仓库自包含加载路径。

看到 `FR3 policy ready: ws://0.0.0.0:8848` 后，机器人连接 `ws://<本机局域网IP>:8848`。`0.0.0.0` 是监听地址。Ctrl+C 停止服务。

## 观测和动作

使用持久连接、JSON 文本帧，一问一答。`obs` 请求示意（图像占位符需替换为实际编码）：

```json
{
  "type": "obs",
  "images": {
    "head_camera": "<base64 PNG/JPEG>",
    "left_wrist_camera": "<base64 PNG/JPEG>"
  },
  "prompt": "Stack the two pink bowls in the center of the table.",
  "state": [0.6, 0.0, 0.3, 1, 0, 0, 0, 1, 0, 1]
}
```

上面的 state 仅展示格式；真机须每次发送当前测量值。

| 字段 | 约定 |
| --- | --- |
| `head_camera` | 必填，头部相机 RGB 图像，编码为 base64 PNG/JPEG，无 data URL 前缀 |
| `left_wrist_camera` | 必填，FR3 腕部相机；单臂也使用此字段名 |
| `right_wrist_camera` | 忽略，始终填黑，与该 checkpoint 的缺失右视图一致 |
| `state` | 10 维、未归一化的实测 `fr3_link8`：`[x,y,z, rot6d(6), gripper]` |
| `prompt` | 可省略或为空，使用默认叠碗指令 |

数据集的两路图像均为 H×W=`480×640`。发送 RGB 原图，OpenCV 的 BGR 帧需先转 RGB。服务端先按训练 reader 用 Pillow **LANCZOS** 将头部缩放为 W×H=`320×256`、腕部缩放为 `160×128`，再拼成 H×W=`384×320`：头部在上方，腕部在左下，右下全黑。客户端无需缩放、裁剪或拼图。

动作响应包含以下字段：

| 字段 | 含义 |
| --- | --- |
| `type` | `"action"` |
| `action` | 默认 `[8,10]` 二维数组，`action[0]` 至 `action[7]` 是按时间排列的 8 个动作，每行 10 个数 |
| `action_count` | 本条响应的动作数，默认 `8` |
| `action_dt_s` | 训练动作间隔，`1/30` 秒 |
| `step` | 本轮成功响应计数，从 `1` 开始，每条响应加 1 |
| `latency_ms` | 本条请求的图像解码/缩放、预处理及推理耗时，不含网络 |

每行 action 已反归一化，顺序同上。服务端使用 `eef_state` 统计归一化状态、映射到 80 维；输出从 80 维取回 10 维，使用 `eef` 统计恢复位置。`FR3Normalizer` 与训练 reader 一样直接复制 rot6d/夹爪分量，不进行额外缩放或裁剪。

物理约定现已由 reader 和数据元数据确认：

- 位置单位为米，位置与姿态均表示 **机器人基坐标系下的 `fr3_link8`**。
- `state` 是当前实测 link8 位姿；`action` 是 **绝对 link8 目标位姿**，不与当前位姿相加。若机器人接口使用不同 TCP/工具坐标系，需在机器人端做相应固定变换。
- rot6d 是旋转矩阵前两列依次拼接：`[R00,R10,R20,R01,R11,R21]`。实测 state 的两列须单位正交（容差 `1e-4`），全零 rot6d 会被拒绝。预测 rot6d 保留模型输出；机器人端转为姿态时需先正交化，再按控制器要求转换为矩阵或四元数。
- 夹爪为 `-1=闭合、+1=张开`，实测 state 须在 `[-1,1]`（容差 `1e-5`）。服务端保留连续预测值；实体宽度标定及控制器接受的夹爪命令转换由机器人端完成。

其他消息：`{"type":"ping"}` 返回 `pong`，包含 `protocol_version=2`、`actions_per_request=8`、`action_shape=[8,10]`、`replan_every_request=true`，以及位姿和频率元数据。`{"type":"reset"}` 返回 `reset_ack` 并重置 policy 状态和请求计数。新连接、断开、任务文本变化及观测/推理错误也会重置状态；第二个并发连接收到 `server_busy`。错误响应格式为 `{"type":"error","code":"...","message":"..."}`，不含动作。

## 动作节奏与客户端接入

训练动作在 **30 Hz** 网格上排列：观测取 row `t` 的实测状态，动作来自 `t:t+32` 的已确认绝对目标保持命令，视频取 `t+[0,4,...,32]`。

**每条观测请求都会重新推理一次**，模型生成 32 个动作，服务端将前 8 个作为一个完整响应返回，其余丢弃。下一条请求使用新的观测生成新计划，服务端不跨请求保留待执行动作。

`--inference-horizon 8` 现在表示**每条响应返回 8 个动作**，也可写为 `--actions-per-request 8`。设置为 `1` 或 `32` 时，每条响应分别返回 1 个或 32 个动作；`action` 始终为二维数组，包括 `[1,10]`。该参数不再表示隔多少条请求才重规划。

机器人端接收一段后，以训练的 30 Hz 节奏依次执行；8 个动作覆盖约 0.267 秒，执行后采集新观测再请求下一段。`--denoise-steps` 控制扩散步数。当前同步服务每条请求都等待完整推理，**不保证墙钟时间上的连续 30 Hz 更新**；机器人控制端需要负责执行时序及推理等待期间的控制。默认关闭 torch.compile 和 DiT 近似缓存，可分别用 `--compile`、`--dit-cache` 开启。

机器人端可复用现有工具，核心调用如下（`head_rgb`、`wrist_rgb` 为 uint8 RGB 数组，`state10` 为当前状态）：

```python
from benchmarks.utils.client import build_payload, encode_numpy_b64
from benchmarks.utils.transport import WSPolicyClient

with WSPolicyClient("ws://<policy主机IP>:8848", timeout=300) as client:
    print(client.ping())
    client.reset()  # 每个 episode 开始时调用
    payload = build_payload(
        head=encode_numpy_b64(head_rgb),
        left_wrist=encode_numpy_b64(wrist_rgb),
        state=state10,
        prompt=task_prompt,
    )
    result = client.predict_once(payload)
    actions = result["action"]  # 默认 8×10；actions[i] 为第 i 个 10 维动作
    dt = result["action_dt_s"]  # 1/30 秒；由机器人端按此间隔依次执行 actions
```

真机循环保持此连接，依次进行“采集观测 → 请求一段动作 → 执行该段 → 采集新观测”。原先按一维 `action` 读取的机器人客户端必须改为遍历二维数组，不能把 8 行当作一个动作。使用 `predict_once` 避免断线时自动重发；超时/断线后关闭旧连接，重新连接并 reset，重新采集观测。

## 联调与检查

先用已上传的数据集做 CPU 一致性检查，并导出首个观测（不会重新计算或覆盖数据集统计量）：

```bash
python scripts/fr3_real_inference/verify_dataset.py
```

脚本核对数据指纹、checkpoint/数据集统计量和默认 prompt，抽查 train/val 共 6 个窗口，逐像素比较首帧拼图，比较状态归一化和动作反归一化。结果保存在 `outputs/fr3_real_inference/verification.json`；同目录生成 `head_camera.png`、`left_wrist_camera.png`、`state.json` 和 `prompt.txt`。

服务启动后，用这些真实观测文件运行原连续测试：

```bash
python scripts/inference_test/inference_continuous_test.py \
  --server ws://127.0.0.1:8848 \
  --head-camera outputs/fr3_real_inference/head_camera.png \
  --left-wrist-camera outputs/fr3_real_inference/left_wrist_camera.png \
  --state-file outputs/fr3_real_inference/state.json \
  --prompt 'Stack the two pink bowls in the center of the table.' \
  -n 16 --reset-every 8
```

两个测试客户端已支持二维动作响应；连续测试日志应显示 `action_count=8  action_dim=10`。`-n 16` 表示 16 条观测请求、16 次推理，共返回 128 个动作。

或者使用随机图像和格式有效的示例状态（仅检查通信，返回值不用于真机执行）：

```bash
python scripts/inference_test/inference_continuous_test.py \
  --server ws://127.0.0.1:8848 --test \
  --state 0.6 0 0.3 1 0 0 0 1 0 1 \
  --prompt 'Stack the two pink bowls in the center of the table.' \
  -n 16 --reset-every 8
```

`state.json` 为 10 维 JSON 数组或 `{"state": [...]}`；固定文件的重复请求只验证接口，不构成闭环控制。原测试默认生成 20 维零 state；仅设置 `--state-dim 10` 仍会生成无效的全零旋转，须传入上述有效示例或真实状态。

无需加载模型的 CPU 检查覆盖每请求重新规划、批量动作顺序/数量、相机布局、状态归一化、WebSocket 连接/reset，以及原连续测试脚本的兼容性：

```bash
python -m unittest discover -s scripts/fr3_real_inference -p 'test_*.py' -v
```

`--check-only`、数据集一致性检查和这些 CPU 测试不代表完整权重加载、GPU 推理或真机任务已验证。

## 启动时显示 `Killed`（已修复）

2026-11-05 的内核日志确认该次故障是主机 RAM OOM：Python 的匿名常驻内存达到约 25.6 GiB，本机总内存约 30 GiB、无 swap。旧加载路径先创建完整随机模型，再加载 checkpoint；T5 文本编码器初始化的 FP32 参数加上已有 DiT/VAE 超出了可用内存，尚未进入 GPU 推理。

`openwam/deploy/model_loader.py` 现对自包含的原生 Wan 单/双系统部署使用 `init_empty_weights(include_buffers=False)` 创建参数空结构，然后由现有严格加载逻辑以 `assign=True` 挂载 checkpoint 权重。运行时常量保留真实数据，避免随机权重副本和初始化期间的 FP32 内存峰值。**启动命令不变**，减小 `--denoise-steps` 或 `--inference-horizon` 无法解决这种初始化 OOM。

OOM 修复验证的空结构初始化峰值约 982 MiB；完整加载的 RSS 峰值约 24.3 GiB（包含映射权重文件页），推理后主进程 RSS 约 2.0 GiB，显存约 24.1 GiB。加载回归测试 4 项通过。该次资源记录为 `outputs/fr3_real_inference/deploy_memory.txt`。

当前批量协议也已用 `checkpoint_step_4500.safetensors`、10 步去噪、horizon=8 完成 GPU 验证：

- 真实数据集观测的连续 2 次请求均返回 `action_count=8, action_dim=10`，每次各完成一次完整推理。
- 首次请求约 1558 ms，第二次约 564 ms；这是本次单组输入的结果，不代表稳定延迟或实时 30 Hz。
- 11 项 FR3 服务测试通过，包括批量动作顺序、每请求重规划及原通用单动作接口行为。
- 临时服务监听 `127.0.0.1:18848`，验证后已退出；没有驱动真机。

批量推理日志：`outputs/fr3_real_inference/chunk_deploy_check.log`。
