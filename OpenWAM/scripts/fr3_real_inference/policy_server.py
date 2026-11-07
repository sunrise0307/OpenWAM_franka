"""FR3 stack-bowl policy endpoint; see README.md for the wire contract and commands."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LOGGER = logging.getLogger("fr3_policy")
RAW_DIM = 10
MAX_MESSAGE_BYTES = 32 * 1024 * 1024
CAMERA_LAYOUT = ["observation.images.head_camera", "observation.images.left_wrist_camera", "__missing_right__"]
# assets/data/fr3_data/stack_bowl/meta/tasks.parquet, task_index=0.
DEFAULT_PROMPT = "Stack the two pink bowls in the center of the table."


class RequestError(ValueError):
    """An observation or message that doesn't satisfy the FR3 contract."""


class FR3Normalizer:
    """Preserve rot6d/gripper exactly as FR3Dataset._normalize does.

    Delegate xyz statistics and unified mapping to OpenWAM. The validated FR3
    maps place raw EEF10 at slots 0:10; bypass affine roundoff/clipping for the
    seven coordinates that the training reader explicitly copies unchanged.
    """

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        # Architecture generation also reads the unified action mask metadata.
        return getattr(self.inner, name)

    def normalize(self, values):
        raw = np.asarray(values, dtype=np.float32)
        result = self.inner.normalize(raw)
        result[..., 3:10] = raw[..., 3:10]
        return result

    def unnormalize(self, values):
        raw = np.asarray(values, dtype=np.float32)
        result = self.inner.unnormalize(raw)
        result[..., 3:10] = raw[..., 3:10]
        return result


def inspect_checkpoint(ckpt_dir: Path, ckpt_name: str | None, horizon: int) -> dict:
    """Check local artifacts and the FR3 contract without constructing the model."""
    from omegaconf import OmegaConf

    from openwam.dataloader.fr3 import FR3_CONTRACT

    cfg = OmegaConf.load(ckpt_dir / "config.yaml")
    expected = {
        "dataloader.type": "fr3",
        "dataloader.action_mode": "eef",
        "dataloader.gripper_convention": "minus1_closed_plus1_open",
        "dataloader.normalize_mode": "min-max",
        "dataloader.unify_action": True,
        "dataloader.unify_action_map": ["0-9"],
        "dataloader.unify_state_map": ["0-9"],
        "dataloader.multiview": True,
        "dataloader.camera_layout": CAMERA_LAYOUT,
        "dataloader.num_frames": 33,
        "dataloader.video_stride": 4,
        "dataloader.height": 384,
        "dataloader.width": 320,
        "model.architecture.use_proprioception": True,
        "model.architecture.action_dim": 80,
        "model.architecture.state_dim": 80,
    }
    for key, value in expected.items():
        actual = OmegaConf.select(cfg, key)
        if actual != value:
            raise ValueError(f"This endpoint expects {key}={value!r}, got {actual!r}")

    if ckpt_name is None:
        numbered = [
            (int(p.stem.removeprefix("checkpoint_step_")), p)
            for p in ckpt_dir.glob("checkpoint_step_*.safetensors")
            if p.stem.removeprefix("checkpoint_step_").isdigit()
        ]
        if not numbered:
            raise FileNotFoundError(f"No checkpoint_step_<N>.safetensors in {ckpt_dir}")
        checkpoint = max(numbered)[1]
    else:
        checkpoint = ckpt_dir / ckpt_name
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    tokenizer = ckpt_dir / str(cfg.model.video_backbone.tokenizer.subdir)
    for name in ("tokenizer_config.json", "tokenizer.json", "spiece.model"):
        if not (tokenizer / name).is_file():
            raise FileNotFoundError(tokenizer / name)

    stats = np.load(ckpt_dir / "normalization_stats.npy", allow_pickle=True).item()
    for block in ("eef", "eef_state"):
        for field in ("min", "max", "mean", "std"):
            values = np.asarray(stats[block][field])
            if values.shape != (RAW_DIM,) or not np.isfinite(values).all():
                raise ValueError(f"{block}.{field} must contain {RAW_DIM} finite values")
        if np.any(stats[block]["max"] < stats[block]["min"]):
            raise ValueError(f"{block}: max must be >= min")
        if not (np.all(stats[block]["min"][3:10] == -1) and np.all(stats[block]["max"][3:10] == 1)):
            raise ValueError(f"{block}: FR3 rot6d/gripper must have identity min-max statistics")

    chunk_size = int(cfg.dataloader.num_frames) - 1
    if not 1 <= horizon <= chunk_size:
        raise ValueError(f"--inference-horizon must be between 1 and {chunk_size}")
    stride = int(cfg.dataloader.video_stride)
    if stride <= 0:
        raise ValueError("dataloader.video_stride must be positive")
    return {
        "checkpoint": str(checkpoint.resolve()),
        "representation": "eef",
        **FR3_CONTRACT,
        "pose_reference_frame": "robot_base",
        "position_unit": "m",
        "rotation_representation": "concatenated_matrix_columns_0_1",
        "action_fps": 30,
        "action_dt_s": 1 / 30,
        "state_dim": RAW_DIM,
        "action_dim": RAW_DIM,
        "image_size": [int(cfg.dataloader.height), int(cfg.dataloader.width)],
        "required_cameras": ["head_camera", "left_wrist_camera"],
        "action_chunk_size": chunk_size,
        "video_num_frames": chunk_size // stride + 1,
        "inference_horizon": horizon,
        "protocol_version": 2,
        "actions_per_request": horizon,
        "action_shape": [horizon, RAW_DIM],
        "replan_every_request": True,
    }


class FR3PolicyService:
    """Single robot session around the existing PolicyServer/WAMPolicy stack."""

    def __init__(self, policy, metadata: dict, default_prompt: str = ""):
        self.policy = policy
        self.metadata = metadata
        self.default_prompt = default_prompt
        self._prompt = None
        self._client = None

    def reset(self):
        self.policy.reset()
        self._prompt = None

    def prepare_observation(self, data: dict) -> dict:
        images = data.get("images")
        if not isinstance(images, dict):
            raise RequestError("images must contain head_camera and left_wrist_camera as base64 PNG/JPEG")
        for camera in ("head_camera", "left_wrist_camera"):
            if not isinstance(images.get(camera), str) or not images[camera]:
                raise RequestError(f"images.{camera} must be a non-empty base64 PNG/JPEG string")
        state = data.get("state")
        if not isinstance(state, list) or len(state) != RAW_DIM:
            raise RequestError("state must be a flat list of 10 raw values: xyz(3), rot6d(6), gripper(1)")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in state):
            raise RequestError("state must contain only numbers")
        if not np.isfinite(state).all() or np.any(np.abs(state) > np.finfo(np.float32).max):
            raise RequestError("state must contain finite float32 values")
        raw = np.asarray(state, dtype=np.float32)
        first, second = raw[3:6], raw[6:9]
        if (
            abs(np.linalg.norm(first) - 1) > 1e-4
            or abs(np.linalg.norm(second) - 1) > 1e-4
            or abs(np.dot(first, second)) > 1e-4
        ):
            raise RequestError("state[3:9] must be two orthonormal rotation-matrix columns for measured fr3_link8")
        if abs(raw[9]) > 1.00001:
            raise RequestError("state[9] must be gripper openness in [-1, 1] (-1 closed, +1 open)")
        prompt = data.get("prompt")
        if prompt is None or prompt == "":
            prompt = self.default_prompt
        if not isinstance(prompt, str) or not prompt.strip():
            raise RequestError("Send a non-empty prompt, or start the server with --prompt")

        from PIL import Image

        def decode_camera(camera, size):
            try:
                with Image.open(io.BytesIO(base64.b64decode(images[camera], validate=True))) as image:
                    # LeRobotV3Reader._decode_one_camera -> decode_video_frames
                    # resizes each tile with LANCZOS BEFORE multiview assembly.
                    return image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
            except Exception as exc:
                raise RequestError(f"images.{camera}: cannot decode PNG/JPEG ({exc})") from exc

        return {
            "images": {
                "head_camera": decode_camera("head_camera", (320, 256)),
                "left_wrist_camera": decode_camera("left_wrist_camera", (160, 128)),
                # The FR3 checkpoint was trained with this slot missing. The
                # generic continuous test sends three cameras; discard its third.
                "right_wrist_camera": None,
            },
            "state": state,
            "prompt": prompt,
        }

    def predict(self, data: dict) -> dict:
        from openwam.deploy.obs_preprocess import ObsValidationError

        start = time.monotonic()
        obs = self.prepare_observation(data)
        if self._prompt is not None and obs["prompt"] != self._prompt:
            self.reset()
        num_actions = self.metadata["actions_per_request"]
        try:
            result = self.policy.predict_chunk(obs, num_actions)
        except ObsValidationError as exc:
            raise RequestError(str(exc)) from exc
        self._prompt = obs["prompt"]
        action = np.asarray(result["action"])
        if action.shape != (num_actions, RAW_DIM) or not np.isfinite(action).all():
            raise RuntimeError(
                f"Policy must return finite actions of shape ({num_actions}, {RAW_DIM}), got {action.shape}"
            )
        result["latency_ms"] = round((time.monotonic() - start) * 1000, 2)
        LOGGER.info("step=%d latency_ms=%.2f", result["step"], result["latency_ms"])
        return {
            **result,
            "type": "action",
            "action": action.tolist(),
            "action_count": num_actions,
            "action_dt_s": self.metadata["action_dt_s"],
        }

    def process_message(self, message: str | bytes) -> dict:
        """Called sequentially in a worker thread so inference doesn't block I/O."""
        try:
            try:
                data = json.loads(message)
            except (ValueError, UnicodeError) as exc:
                raise RequestError("Expected a JSON object") from exc
            if not isinstance(data, dict):
                raise RequestError("Expected a JSON object")
            kind = data.get("type", "obs")
            if kind == "ping":
                return {"type": "pong", **self.metadata}
            if kind == "reset":
                self.reset()
                return {"type": "reset_ack"}
            if kind == "obs":
                return self.predict(data)
            return {"type": "error", "code": "unknown_message_type", "message": f"Unknown message type: {kind}"}
        except RequestError as exc:
            self.reset()
            return {"type": "error", "code": "obs_validation_error", "message": str(exc)}
        except Exception as exc:
            LOGGER.exception("Prediction failed; clearing the action buffer")
            self.reset()
            return {"type": "error", "code": "internal_error", "message": str(exc)}

    async def handle_connection(self, websocket):
        from websockets.exceptions import ConnectionClosed

        # A shared action queue belongs to one robot. A second connection must
        # never consume or reset the first robot's trajectory.
        if self._client is not None:
            await websocket.send(
                json.dumps({"type": "error", "code": "server_busy", "message": "One robot is already connected"})
            )
            await websocket.close(code=1013, reason="One robot is already connected")
            return
        self._client = websocket
        LOGGER.info("Client connected: %s", websocket.remote_address)
        try:
            await asyncio.to_thread(self.reset)
            async for message in websocket:
                result = await asyncio.to_thread(self.process_message, message)
                await websocket.send(json.dumps(result, allow_nan=False))
        except ConnectionClosed:
            pass
        finally:
            # Don't release ownership until any in-flight prediction finishes
            # and its remaining actions have been discarded.
            await asyncio.to_thread(self.reset)
            self._client = None
            LOGGER.info("Client disconnected; action buffer cleared")

    async def serve(self, host: str, port: int):
        from websockets.asyncio.server import serve

        async with serve(
            self.handle_connection,
            host,
            port,
            max_size=MAX_MESSAGE_BYTES,
            max_queue=1,
            compression=None,
            ping_interval=None,
        ):
            LOGGER.info("FR3 policy ready: ws://%s:%d", host, port)
            await asyncio.Future()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, default=ROOT / "assets/fr3_stack_bowl_ckpt")
    parser.add_argument("--ckpt-name", default=None, help="Default: highest checkpoint_step_N")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8848)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT, help="Fallback instruction (default: recorded stack-bowl task)"
    )
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument(
        "--inference-horizon",
        "--actions-per-request",
        dest="inference_horizon",
        type=int,
        default=8,
        help="Actions returned per observation (1..32, default 8); every request replans",
    )
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (slower first inference)")
    parser.add_argument("--dit-cache", action="store_true", help="Enable approximate DiT velocity caching")
    parser.add_argument(
        "--check-only", action="store_true", help="Check config/stats/artifacts without loading weights"
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    if args.denoise_steps < 1 or not 1 <= args.port <= 65535:
        parser.error("--denoise-steps must be positive and --port must be in 1..65535")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ckpt_dir = args.ckpt_dir.expanduser().resolve()
    metadata = inspect_checkpoint(ckpt_dir, args.ckpt_name, args.inference_horizon)
    metadata["default_prompt"] = args.prompt
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    if args.check_only:
        return

    from omegaconf import OmegaConf

    from openwam.deploy.server import build_server_from_config

    deploy_cfg = OmegaConf.load(ROOT / "configs/deploy.yaml")
    deploy_cfg.inference.denoise_steps = args.denoise_steps
    deploy_cfg.inference.denoise_mode = "sync"
    deploy_cfg.inference.inference_mode = "sync"
    deploy_cfg.inference.inference_horizon = args.inference_horizon
    deploy_cfg.inference.inference_delay_steps = None
    deploy_cfg.optimization.decode_video = False
    deploy_cfg.optimization.compile.enabled = args.compile
    deploy_cfg.optimization.dit_cache.enabled = args.dit_cache
    policy = build_server_from_config(
        deploy_cfg,
        str(ckpt_dir),
        device=args.device,
        ckpt_name=metadata["checkpoint"],
    )
    architecture = policy.engine.architecture
    architecture.attach_normalizer(FR3Normalizer(architecture.normalizer))
    service = FR3PolicyService(policy, metadata, args.prompt)
    try:
        asyncio.run(service.serve(args.host, args.port))
    except KeyboardInterrupt:
        LOGGER.info("Stopping FR3 policy server")
    finally:
        policy.shutdown()


if __name__ == "__main__":
    main()
