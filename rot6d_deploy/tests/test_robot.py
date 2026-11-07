import math
import threading
import unittest

import numpy as np

from rot6d_deploy.control.geometry import (
    RECORDED_GRIPPER_MAX_WIDTH_M,
    gripper_fraction_to_width,
    gripper_width_to_fraction,
    quaternion_distance,
    quaternion_to_rotation_six,
)
from rot6d_deploy.control.robot import RobotController, SafetyLimits


class FakeArm:
    def __init__(self):
        self.parameters = None

    def update_current_policy(self, parameters):
        self.parameters = parameters
        return 17


class RobotTests(unittest.TestCase):
    def test_recorded_gripper_scale(self):
        self.assertEqual(gripper_width_to_fraction(RECORDED_GRIPPER_MAX_WIDTH_M), 1.0)
        self.assertAlmostEqual(gripper_fraction_to_width(0.25), RECORDED_GRIPPER_MAX_WIDTH_M / 4)

    def test_full_absolute_target_is_sent_without_client_truncation(self):
        controller = RobotController.__new__(RobotController)
        controller.limits = SafetyLimits((0.3, -0.2, 0.1), (0.8, 0.2, 0.7))
        controller._read_pose = lambda: (np.array([0.5, 0.0, 0.3]), np.array([0, 0, 0, 1]))
        controller._lock = threading.RLock()
        controller._gripper_condition = threading.Condition()
        controller._active = True
        controller._closed = False
        controller._fault = None
        controller._sequence = 0
        controller._last_gripper_target = 1.0
        controller.arm = FakeArm()

        orientation = [0, 0, math.sin(0.03), math.cos(0.03)]
        action = [0.5209, 0, 0.3, *quaternion_to_rotation_six(orientation), 1]
        self.assertEqual(controller.apply_action(action), 17)
        parameters = controller.arm.parameters
        np.testing.assert_allclose(parameters["ee_pos_desired"].numpy(), action[:3], atol=1e-7)
        self.assertLess(quaternion_distance(parameters["ee_quat_desired"].numpy(), orientation), 1e-5)


if __name__ == "__main__":
    unittest.main()
