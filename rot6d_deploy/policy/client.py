"""One observation receives one complete policy action group."""

import base64
import inspect
import io
import json
import math

import numpy as np


class PolicyError(RuntimeError):
    pass


def encode_rgb_image(image):
    from PIL import Image

    value = np.asarray(image)
    if value.shape != (480, 640, 3) or value.dtype != np.uint8:
        raise ValueError("camera image must be 480 by 640 RGB bytes")
    buffer = io.BytesIO()
    Image.fromarray(value, mode="RGB").save(buffer, format="PNG", compress_level=1)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_observation(head_image, wrist_image, state, prompt):
    values = np.asarray(state, dtype=np.float64)
    if values.shape != (10,) or not np.isfinite(values).all():
        raise ValueError("measured state must contain ten finite numbers")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("task instruction must be nonempty")
    return {
        "type": "obs",
        "images": {
            "head_camera": encode_rgb_image(head_image),
            "left_wrist_camera": encode_rgb_image(wrist_image),
        },
        "state": values.tolist(),
        "prompt": prompt,
    }


def validate_metadata(info, expected_frequency):
    if not isinstance(info, dict) or info.get("type") != "pong":
        raise PolicyError("policy server did not return its metadata")
    expected = {
        "action_dim": 10,
        "state_dim": 10,
        "action_representation": "absolute_target_eef10",
        "action_frame": "fr3_link8",
        "pose_reference_frame": "robot_base",
        "rotation_representation": "concatenated_matrix_columns_0_1",
    }
    for key, value in expected.items():
        if info.get(key) != value:
            raise PolicyError("policy metadata %s is incompatible: %r" % (key, info.get(key)))
    frequency = info.get("action_fps")
    if isinstance(frequency, bool) or not isinstance(frequency, (float, int)) or not math.isfinite(frequency):
        raise PolicyError("policy action frequency is missing")
    if abs(frequency - expected_frequency) > 1e-6:
        raise PolicyError("policy action frequency differs from the requested frequency")
    interval = info.get("action_dt_s")
    if (isinstance(interval, bool) or not isinstance(interval, (float, int)) or
            not math.isfinite(interval) or interval <= 0 or
            not math.isclose(interval * frequency, 1.0, rel_tol=1e-6, abs_tol=1e-6)):
        raise PolicyError("policy action interval is incompatible with its frequency")
    count = info.get("actions_per_request")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise PolicyError("policy action count must be a positive integer")
    if info.get("action_shape") != [count, expected["action_dim"]]:
        raise PolicyError("policy action shape does not match its action count")
    return info


class PolicyClient:
    def __init__(self, url, expected_frequency=30.0, timeout_s=30.0, open_timeout_s=5.0):
        if not url.startswith(("ws://", "wss://")):
            raise ValueError("policy address must use WebSocket")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("policy timeout must be positive")
        self.url = url
        self.expected_frequency = expected_frequency
        self.timeout_s = timeout_s
        self.open_timeout_s = open_timeout_s
        self.socket = None
        self.expected_action_count = None

    def connect(self):
        from websockets.sync.client import connect

        if self.socket is not None:
            raise RuntimeError("policy socket is already connected")
        arguments = {
            "open_timeout": self.open_timeout_s,
            "max_size": 2 * 1024 * 1024,
            "compression": None,
            "ping_interval": None,
            "proxy": None,
        }
        supported = inspect.signature(connect).parameters
        self.socket = connect(self.url, **{key: value for key, value in arguments.items() if key in supported})
        try:
            metadata = validate_metadata(self._exchange({"type": "ping"}), self.expected_frequency)
            response = self._exchange({"type": "reset"})
            if response.get("type") != "reset_ack":
                raise PolicyError("policy server did not acknowledge session reset")
            self.expected_action_count = metadata["actions_per_request"]
            return metadata
        except Exception:
            self.close()
            raise

    def _exchange(self, request):
        if self.socket is None:
            raise PolicyError("policy socket is disconnected")
        try:
            self.socket.send(json.dumps(request, allow_nan=False, separators=(",", ":")))
            raw = self.socket.recv(timeout=self.timeout_s)
            result = json.loads(raw)
        except Exception as exc:
            self.close()
            raise PolicyError("policy request failed: %s" % exc) from exc
        if not isinstance(result, dict):
            self.close()
            raise PolicyError("policy response is not an object")
        if result.get("type") == "error":
            self.close()
            raise PolicyError("policy server error %s: %s" % (result.get("code"), result.get("message")))
        return result

    def predict_batch(self, observation):
        """Read all targets from one response to one observation."""
        response = self._exchange(observation)
        if response.get("type") != "action":
            self.close()
            raise PolicyError("policy response is not an action")
        count = response.get("action_count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            self.close()
            raise PolicyError("policy response must declare a positive action count")
        if self.expected_action_count is not None and count != self.expected_action_count:
            self.close()
            raise PolicyError("policy response action count differs from server metadata")
        try:
            actions = np.asarray(response.get("action"), dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            self.close()
            raise PolicyError("policy actions do not contain numbers") from exc
        if actions.shape != (count, 10) or not np.isfinite(actions).all():
            self.close()
            raise PolicyError("policy response shape or action values are invalid")
        return actions

    def close(self):
        socket, self.socket = self.socket, None
        self.expected_action_count = None
        if socket is not None:
            socket.close()
