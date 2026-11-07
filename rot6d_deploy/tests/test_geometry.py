import unittest

import numpy as np

from rot6d_deploy.control.geometry import (
    measured_state,
    parse_action,
    quaternion_distance,
    quaternion_to_rotation_six,
    rotation_six_to_matrix,
    rotation_six_to_quaternion_xyzw,
)
from rot6d_deploy.control.robot import SafetyLimits


class GeometryTests(unittest.TestCase):
    def test_identity_and_half_turns(self):
        cases = (
            ([1, 0, 0, 0, 1, 0], [0, 0, 0, 1]),
            ([1, 0, 0, 0, -1, 0], [1, 0, 0, 0]),
            ([-1, 0, 0, 0, 1, 0], [0, 1, 0, 0]),
        )
        for rotation, expected in cases:
            with self.subTest(rotation=rotation):
                quaternion = rotation_six_to_quaternion_xyzw(rotation)
                self.assertLess(quaternion_distance(quaternion, expected), 1e-6)

    def test_orthogonalization_and_round_trip(self):
        original = np.array([0.2, -0.3, 0.1, 0.9])
        original /= np.linalg.norm(original)
        rotation = quaternion_to_rotation_six(original)
        matrix = rotation_six_to_matrix(rotation + [0.02, 0, 0, 0, 0.02, 0])
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
        self.assertLess(quaternion_distance(rotation_six_to_quaternion_xyzw(rotation), original), 1e-6)

    def test_degenerate_rotation_is_rejected(self):
        for rotation in ([0] * 6, [1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 1e-10, 0]):
            with self.subTest(rotation=rotation), self.assertRaises(ValueError):
                rotation_six_to_quaternion_xyzw(rotation)

    def test_absolute_target_and_gripper_convention(self):
        action = [0.6, 0.1, 0.3, 1, 0, 0, 0, 1, 0, -1]
        target = parse_action(action)
        self.assertEqual(target.gripper_open_fraction, 0.0)
        self.assertEqual(measured_state([0.6, 0.1, 0.3], [0, 0, 0, 1], 1.0)[9], 1.0)
        np.testing.assert_allclose(target.position, [0.6, 0.1, 0.3])
        self.assertLess(quaternion_distance(target.quaternion_xyzw, [0, 0, 0, 1]), 1e-6)

    def test_workspace_bounds_must_be_ordered(self):
        with self.assertRaises(ValueError):
            SafetyLimits((0, 0, 0), (0, 1, 1))

    def test_configured_workspace_accepts_recording_range(self):
        limits = SafetyLimits((-0.858, -0.858, -0.346), (0.858, 0.858, 1.191))
        self.assertEqual(limits.workspace_min[2], -0.346)
        self.assertEqual(limits.workspace_max[2], 1.191)


if __name__ == "__main__":
    unittest.main()
