import json
import threading
import unittest

import numpy as np

from rot6d_deploy.policy.client import (
    PolicyClient,
    PolicyError,
    build_observation,
    validate_metadata,
)


class FakeSocket:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.sent = []
        self.closed = False

    def send(self, text):
        self.sent.append(json.loads(text))

    def recv(self, timeout):
        return json.dumps(next(self.responses))

    def close(self):
        self.closed = True


class PolicyTests(unittest.TestCase):
    @staticmethod
    def metadata():
        return {
            "type": "pong", "action_dim": 10, "state_dim": 10,
            "action_representation": "absolute_target_eef10",
            "action_frame": "fr3_link8", "pose_reference_frame": "robot_base",
            "rotation_representation": "concatenated_matrix_columns_0_1",
            "action_fps": 30, "action_dt_s": 1 / 30,
            "actions_per_request": 8, "action_shape": [8, 10],
        }

    def test_metadata_rejects_wrong_pose_frame(self):
        metadata = self.metadata()
        validate_metadata(metadata, 30)
        metadata["pose_reference_frame"] = "world"
        with self.assertRaises(PolicyError):
            validate_metadata(metadata, 30)

    def test_metadata_accepts_variable_action_counts(self):
        for count in (1, 8, 32):
            with self.subTest(count=count):
                metadata = self.metadata()
                metadata["actions_per_request"] = count
                metadata["action_shape"] = [count, 10]
                self.assertEqual(validate_metadata(metadata, 30), metadata)

    def test_metadata_rejects_action_count_shape_mismatch(self):
        metadata = self.metadata()
        metadata["actions_per_request"] = 3
        with self.assertRaises(PolicyError):
            validate_metadata(metadata, 30)

    def test_wrong_count_shape_or_values_close_connection(self):
        action = [0.5, 0, 0.3, 1, 0, 0, 0, 1, 0, 1]
        responses = [
            {"type": "action", "action_count": 1, "action": [action] * 8},
            {"type": "action", "action_count": 0, "action": []},
            {"type": "action", "action_count": 8, "action": action},
            {"type": "action", "action_count": 8, "action": [action] * 7 + [[float("nan")] * 10]},
            {"type": "action", "action_count": True, "action": [action] * 8},
        ]
        for response in responses:
            with self.subTest(response=response):
                socket = FakeSocket([response])
                client = PolicyClient("ws://example.invalid")
                client.socket = socket
                with self.assertRaises(PolicyError):
                    client.predict_batch({"type": "obs"})
                self.assertTrue(socket.closed)
                self.assertEqual(len(socket.sent), 1)

    def test_error_response_is_not_retried(self):
        socket = FakeSocket([{"type": "error", "code": "internal_error", "message": "failed"}])
        client = PolicyClient("ws://example.invalid")
        client.socket = socket
        with self.assertRaises(PolicyError):
            client.predict_batch({"type": "obs"})
        self.assertEqual(len(socket.sent), 1)
        self.assertTrue(socket.closed)

    def test_response_count_must_match_server_metadata(self):
        action = [0.5, 0, 0.3, 1, 0, 0, 0, 1, 0, 1]
        socket = FakeSocket([{"type": "action", "action_count": 2,
                              "action": [action] * 2}])
        client = PolicyClient("ws://example.invalid")
        client.socket = socket
        client.expected_action_count = 3
        with self.assertRaises(PolicyError):
            client.predict_batch({"type": "obs"})
        self.assertTrue(socket.closed)

    def test_one_reply_preserves_variable_numbers_of_ordered_actions(self):
        for count in (1, 8, 32):
            with self.subTest(count=count):
                actions = [[0.5 + step / 1000, 0, 0.3, 1, 0, 0, 0, 1, 0, 1]
                           for step in range(1, count + 1)]
                socket = FakeSocket([{"type": "action", "action_count": count,
                                      "action": actions}])
                client = PolicyClient("ws://example.invalid")
                client.socket = socket
                client.expected_action_count = count
                result = client.predict_batch({"type": "obs"})
                self.assertEqual(result.shape, (count, 10))
                np.testing.assert_allclose(result, actions)
                self.assertEqual(socket.sent, [{"type": "obs"}])

    def test_observation_contains_two_encoded_cameras(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        payload = build_observation(image, image, [0.5, 0, 0.3, 1, 0, 0, 0, 1, 0, 1], "stack bowls")
        self.assertEqual(payload["type"], "obs")
        self.assertEqual(set(payload["images"]), {"head_camera", "left_wrist_camera"})
        self.assertEqual(len(payload["state"]), 10)

    def test_real_websocket_session_reset_and_one_action_group(self):
        from websockets.sync.server import serve

        action = [0.5, 0, 0.3, 1, 0, 0, 0, 1, 0, 1]
        received = []

        def handle(socket):
            for message in socket:
                request = json.loads(message)
                received.append(request["type"])
                if request["type"] == "ping":
                    metadata = self.metadata()
                    metadata["actions_per_request"] = 3
                    metadata["action_shape"] = [3, 10]
                    socket.send(json.dumps(metadata))
                elif request["type"] == "reset":
                    socket.send(json.dumps({"type": "reset_ack"}))
                else:
                    socket.send(json.dumps({"type": "action", "action_count": 3,
                                            "action": [action] * 3}))

        with serve(handle, "127.0.0.1", 0) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.socket.getsockname()[1]
            client = PolicyClient("ws://127.0.0.1:%d" % port, timeout_s=2)
            try:
                client.connect()
                np.testing.assert_allclose(client.predict_batch({"type": "obs"}), [action] * 3)
            finally:
                client.close()
                server.shutdown()
                thread.join(timeout=2)
        self.assertEqual(received, ["ping", "reset", "obs"])


if __name__ == "__main__":
    unittest.main()
