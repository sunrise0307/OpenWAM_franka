"""CPU tests with a synthetic engine; no checkpoint tensors or robot are used."""

import asyncio
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from omegaconf import OmegaConf  # noqa: E402
from policy_server import DEFAULT_PROMPT, FR3Normalizer, FR3PolicyService, inspect_checkpoint  # noqa: E402
from websockets.asyncio.client import connect  # noqa: E402
from websockets.asyncio.server import serve  # noqa: E402

from benchmarks.utils.client import build_payload, encode_numpy_b64  # noqa: E402
from benchmarks.utils.transport import WSPolicyClient  # noqa: E402
from openwam.deploy.model_loader import _build_normalizer  # noqa: E402
from openwam.deploy.server import PolicyServer, merge_deploy_cfg  # noqa: E402

CKPT = ROOT / "assets/fr3_stack_bowl_ckpt"


class SyntheticEngine:
    """Deterministic action chunks make cache reuse/replanning observable."""

    architecture = SimpleNamespace(binary_command_dims=())

    def __init__(self):
        self.calls = []

    def generate(self, conditions):
        self.calls.append(conditions)
        # Distinct values per plan AND per step detect stale, duplicated, or
        # reordered actions in a response without loading real weights.
        values = 100 * len(self.calls) + np.arange(32, dtype=np.float32)
        return {"actions": np.repeat(values[:, None], 10, axis=1)}


def expected_chunk(call, horizon=8):
    return [[float(100 * call + step)] * 10 for step in range(horizon)]


def make_service(horizon=8):
    cfg = merge_deploy_cfg(
        OmegaConf.load(CKPT / "config.yaml"),
        OmegaConf.create({"inference": {"inference_mode": "sync", "inference_horizon": horizon}}),
    )
    engine = SyntheticEngine()
    policy = PolicyServer(engine, cfg)
    return FR3PolicyService(policy, inspect_checkpoint(CKPT, None, horizon)), engine


def observation(prompt="synthetic protocol test"):
    rgb = encode_numpy_b64(np.full((48, 64, 3), 200, dtype=np.uint8))
    return build_payload(
        head=rgb,
        left_wrist=rgb,
        right_wrist=rgb,
        prompt=prompt,
        state=[0.6, 0, 0.3, 1, 0, 0, 0, 1, 0, 1],
    )


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.service, self.engine = make_service()
        self.addCleanup(self.service.policy.shutdown)

    def request(self, obs):
        return self.service.process_message(json.dumps(obs))

    def test_fresh_ordered_chunk_views_and_prompt_change(self):
        result = self.request(observation())
        self.assertEqual(result["action"], expected_chunk(1))
        self.assertEqual(result["action_count"], 8)
        self.assertEqual(result["action_dt_s"], 1 / 30)
        next_obs = observation()
        next_obs["state"][0] = 0.61
        result = self.request(next_obs)
        self.assertEqual(result["action"], expected_chunk(2))
        self.assertEqual(result["step"], 2)
        self.assertEqual(len(self.engine.calls), 2)
        self.assertAlmostEqual(self.engine.calls[1]["proprio"][0], 0.61)
        frame = np.asarray(self.engine.calls[0]["first_frame_image"][0])
        self.assertEqual(frame.shape, (384, 320, 3))
        self.assertTrue((frame[:256] == 200).all())
        self.assertTrue((frame[256:, :160] == 200).all())
        self.assertTrue((frame[256:, 160:] == 0).all())
        result = self.request(observation("another synthetic task"))
        self.assertEqual(result["step"], 1)
        self.assertEqual(result["action"], expected_chunk(3))

    def test_configured_chunk_size_one_and_full_horizon(self):
        for horizon in (1, 32):
            with self.subTest(horizon=horizon):
                service, engine = make_service(horizon)
                try:
                    result = service.process_message(json.dumps(observation()))
                    self.assertEqual(result["action"], expected_chunk(1, horizon))
                    self.assertEqual(result["action_count"], horizon)
                    self.assertEqual(len(engine.calls), 1)
                finally:
                    service.policy.shutdown()

    def test_generic_single_action_api_keeps_existing_buffer_behavior(self):
        for step in range(2):
            obs = self.service.prepare_observation(observation())
            result = self.service.policy.predict(obs)
            self.assertEqual(result["action"], [100.0 + step] * 10)
        self.assertEqual(len(self.engine.calls), 1)

    def test_wrong_state_width_clears_buffer(self):
        self.request(observation())
        bad = observation()
        bad["state"] = [0.0] * 20  # The reference client's default isn't FR3's width.
        self.assertEqual(self.request(bad)["code"], "obs_validation_error")
        self.assertEqual(len(self.engine.calls), 1)
        self.assertEqual(self.request(observation())["action"], expected_chunk(2))

    def test_required_wrist_and_default_prompt(self):
        obs = observation("")
        self.assertEqual(self.request(obs)["code"], "obs_validation_error")
        self.service.default_prompt = "fallback task instruction"
        self.assertEqual(self.request(obs)["type"], "action")
        self.assertEqual(self.engine.calls[0]["prompt"], "fallback task instruction")
        obs["images"]["left_wrist_camera"] = None
        self.assertEqual(self.request(obs)["code"], "obs_validation_error")

    def test_measured_rotation_and_gripper_contract(self):
        for rotation, gripper in (([0] * 6, 0), ([1, 0, 0, 1, 0, 0], 0), ([1, 0, 0, 0, 1, 0], 2)):
            with self.subTest(rotation=rotation, gripper=gripper):
                obs = observation()
                obs["state"][3:9] = rotation
                obs["state"][9] = gripper
                self.assertEqual(self.request(obs)["code"], "obs_validation_error")
        self.assertEqual(len(self.engine.calls), 0)

    def test_checkpoint_directional_normalization(self):
        cfg = OmegaConf.load(CKPT / "config.yaml")
        normalizer = FR3Normalizer(_build_normalizer(cfg, str(CKPT)))
        stats = np.load(CKPT / "normalization_stats.npy", allow_pickle=True).item()
        state = (stats["eef_state"]["min"] + stats["eef_state"]["max"]) / 2
        normalized = normalizer.normalize(state)
        self.assertEqual(normalized.shape, (80,))
        np.testing.assert_allclose(normalized, np.zeros(80), atol=1e-6)
        physical = normalizer.unnormalize(np.zeros((1, 80), dtype=np.float32))
        self.assertEqual(physical.shape, (1, 10))
        np.testing.assert_allclose(physical[0], (stats["eef"]["min"] + stats["eef"]["max"]) / 2)
        # Match the reader's explicit passthrough, including float32 roundoff
        # near unit rotation columns and continuous gripper observations.
        state[3:10] = [0.65970367, 0.7475713, 0.07699511, 0.7468694, -0.6635534, 0.04339271, 0.1234567]
        np.testing.assert_array_equal(normalizer.normalize(state)[3:10], state[3:10])
        np.testing.assert_array_equal(normalizer._dst_index, np.arange(10))

    def test_advertised_pose_and_time_contract(self):
        meta = self.request({"type": "ping"})
        self.assertEqual(meta["action_representation"], "absolute_target_eef10")
        self.assertEqual(meta["action_frame"], "fr3_link8")
        self.assertEqual(meta["pose_reference_frame"], "robot_base")
        self.assertEqual(meta["action_fps"], 30)
        self.assertEqual(meta["protocol_version"], 2)
        self.assertEqual(meta["action_shape"], [8, 10])
        self.assertEqual(meta["actions_per_request"], 8)
        self.assertTrue(meta["replan_every_request"])


class WebSocketTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service, self.engine = make_service()
        self.server = await serve(self.service.handle_connection, "127.0.0.1", 0, ping_interval=None)
        self.url = f"ws://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"

    async def asyncTearDown(self):
        self.server.close()
        await self.server.wait_closed()
        self.service.policy.shutdown()

    async def test_existing_sync_client(self):
        def roundtrip():
            with WSPolicyClient(self.url) as client:
                self.assertEqual(client.ping()["state_dim"], 10)
                self.assertEqual(client.reset()["type"], "reset_ack")
                self.assertEqual(client.predict_once(observation())["action"], expected_chunk(1))
                self.assertEqual(client.reset()["type"], "reset_ack")
                self.assertEqual(client.predict_once(observation())["action"], expected_chunk(2))

        await asyncio.to_thread(roundtrip)

    async def test_single_owner_and_disconnect_reset(self):
        async with connect(self.url, proxy=None) as first:
            await first.send(json.dumps(observation()))
            self.assertEqual(json.loads(await first.recv())["action"], expected_chunk(1))
            async with connect(self.url, proxy=None) as second:
                self.assertEqual(json.loads(await second.recv())["code"], "server_busy")
            await first.send(json.dumps(observation()))
            self.assertEqual(json.loads(await first.recv())["action"], expected_chunk(2))

        # Wait for server-side session cleanup, bounded by the test timeout.
        async def released():
            while self.service._client is not None:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(released(), timeout=3)
        async with connect(self.url, proxy=None) as client:
            await client.send(json.dumps(observation()))
            result = json.loads(await client.recv())
            self.assertEqual(result["step"], 1)
            self.assertEqual(result["action"], expected_chunk(3))

    async def test_original_continuous_script(self):
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(ROOT / "scripts/inference_test/inference_continuous_test.py"),
            "--server",
            self.url,
            "--test",
            "--state",
            *(str(x) for x in observation()["state"]),
            "--prompt",
            DEFAULT_PROMPT,
            "-n",
            "4",
            "--reset-every",
            "2",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            output, _ = await asyncio.wait_for(process.communicate(), timeout=30)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        self.assertEqual(process.returncode, 0, output.decode())
        self.assertIn("done: 4/4 ok", output.decode())
        self.assertIn("action_count=8  action_dim=10", output.decode())
        self.assertEqual(len(self.engine.calls), 4)


if __name__ == "__main__":
    unittest.main()
