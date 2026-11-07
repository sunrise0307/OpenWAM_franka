"""Strict conversion between the policy pose and the robot pose."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PoseTarget:
    position: np.ndarray
    quaternion_xyzw: np.ndarray
    gripper_open_fraction: float


RECORDED_GRIPPER_MAX_WIDTH_M = 0.07982299476861954


def _finite_vector(value, length, name):
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError("%s must contain %d finite numbers" % (name, length))
    return vector


def unit_quaternion(value):
    quaternion = _finite_vector(value, 4, "quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        raise ValueError("quaternion has zero length")
    return quaternion / norm


def rotation_six_to_matrix(value):
    columns = _finite_vector(value, 6, "six-value rotation")
    first = columns[:3]
    first_length = float(np.linalg.norm(first))
    if first_length < 1e-6:
        raise ValueError("first rotation column is degenerate")
    first = first / first_length
    second = columns[3:] - np.dot(first, columns[3:]) * first
    second_length = float(np.linalg.norm(second))
    if second_length < 1e-6:
        raise ValueError("second rotation column is degenerate")
    second = second / second_length
    return np.column_stack((first, second, np.cross(first, second)))


def matrix_to_quaternion_xyzw(matrix):
    """Return a unit quaternion in the ordering expected by Polymetis."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation matrix must be finite and three by three")
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        result = np.array([
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            0.25 * scale,
        ])
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
            result = np.array([0.25 * scale,
                               (matrix[0, 1] + matrix[1, 0]) / scale,
                               (matrix[0, 2] + matrix[2, 0]) / scale,
                               (matrix[2, 1] - matrix[1, 2]) / scale])
        elif index == 1:
            scale = math.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
            result = np.array([(matrix[0, 1] + matrix[1, 0]) / scale,
                               0.25 * scale,
                               (matrix[1, 2] + matrix[2, 1]) / scale,
                               (matrix[0, 2] - matrix[2, 0]) / scale])
        else:
            scale = math.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
            result = np.array([(matrix[0, 2] + matrix[2, 0]) / scale,
                               (matrix[1, 2] + matrix[2, 1]) / scale,
                               0.25 * scale,
                               (matrix[1, 0] - matrix[0, 1]) / scale])
    result = unit_quaternion(result)
    return -result if result[3] < 0 else result


def rotation_six_to_quaternion_xyzw(value):
    return matrix_to_quaternion_xyzw(rotation_six_to_matrix(value))


def quaternion_to_rotation_six(value):
    x, y, z, w = unit_quaternion(value)
    matrix = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return np.concatenate((matrix[:, 0], matrix[:, 1]))


def quaternion_distance(first, second):
    dot = abs(float(np.dot(unit_quaternion(first), unit_quaternion(second))))
    return 2 * math.acos(min(1.0, max(-1.0, dot)))


def parse_action(value):
    action = _finite_vector(value, 10, "policy action")
    if abs(float(action[9])) > 1.25:
        raise ValueError("gripper target is outside the accepted model range")
    return PoseTarget(
        position=action[:3].copy(),
        quaternion_xyzw=rotation_six_to_quaternion_xyzw(action[3:9]),
        gripper_open_fraction=float(np.clip((action[9] + 1) / 2, 0, 1)),
    )


def gripper_width_to_fraction(width_m, maximum_width_m=RECORDED_GRIPPER_MAX_WIDTH_M):
    width = float(width_m)
    maximum = float(maximum_width_m)
    if not math.isfinite(width) or not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("gripper width and calibrated maximum must be finite")
    return float(np.clip(width / maximum, 0, 1))


def gripper_fraction_to_width(fraction, maximum_width_m=RECORDED_GRIPPER_MAX_WIDTH_M):
    value = float(fraction)
    maximum = float(maximum_width_m)
    if not math.isfinite(value) or not 0 <= value <= 1 or not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("gripper fraction and calibrated maximum are invalid")
    return value * maximum


def measured_state(position, quaternion_xyzw, gripper_open_fraction):
    position = _finite_vector(position, 3, "measured position")
    rotation = quaternion_to_rotation_six(quaternion_xyzw)
    fraction = float(gripper_open_fraction)
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("measured gripper openness must be between zero and one")
    return np.concatenate((position, rotation, [2 * fraction - 1])).tolist()
