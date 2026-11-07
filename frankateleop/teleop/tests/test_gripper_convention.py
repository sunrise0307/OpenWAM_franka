import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from polymetis.gripper_interface import GripperInterface
from teleop.robots.dynamixel import DynamixelRobot
from teleop.robots.fr3 import fr3Robot


class GripperConventionTest(unittest.TestCase):
    def test_physical_width_and_normalized_openness(self):
        gripper = GripperInterface.__new__(GripperInterface)
        gripper.metadata = SimpleNamespace(max_width=0.08)
        gripper.get_state = Mock()
        gripper.goto = Mock()

        for width, expected in ((0.0, 0.0), (0.04, 0.5), (0.08, 1.0)):
            gripper.get_state.return_value = SimpleNamespace(width=width)
            self.assertAlmostEqual(gripper.get_open_fraction(), expected)
            gripper.goto_open_fraction(expected, speed=0.1, force=1.0)
            self.assertAlmostEqual(gripper.goto.call_args.kwargs["width"], width)

        self.assertEqual(gripper.goto.call_count, 3)
        for invalid in (-0.1, 1.1, math.nan):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    gripper.goto_open_fraction(invalid, speed=0.1, force=1.0)
        self.assertEqual(gripper.goto.call_count, 3)

    def test_hand_controller_to_fr3_command(self):
        leader = DynamixelRobot.__new__(DynamixelRobot)
        leader._joint_ids = tuple(range(8))
        leader._joint_offsets = np.zeros(8)
        leader._joint_signs = np.ones(8)
        leader._driver = Mock()
        leader._last_pos = None
        leader._alpha = 0.99
        leader.gripper_open_close = tuple(np.deg2rad([200.30, 158.50]))

        follower = fr3Robot.__new__(fr3Robot)
        follower.robot = Mock()
        follower.gripper = Mock()

        for angle, expected in ((158.50, 0.0), (179.40, 0.5), (200.30, 1.0)):
            leader._last_pos = None
            leader._driver.get_joints.return_value = np.array(
                [0.0] * 7 + [np.deg2rad(angle)]
            )
            action = leader.get_joint_state()
            self.assertAlmostEqual(action[-1], expected)
            follower.command_joint_state(action)
            call = follower.gripper.goto_open_fraction.call_args
            self.assertAlmostEqual(call.args[0], expected)
            self.assertEqual(call.kwargs, {"speed": 1, "force": 1})

    def test_fr3_observation_uses_normalized_openness(self):
        robot = fr3Robot.__new__(fr3Robot)
        robot.robot = Mock()
        robot.robot.get_joint_positions.return_value = torch.zeros(7)
        robot.robot.get_ee_pose.return_value = (
            torch.zeros(3), torch.tensor([0.0, 0.0, 0.0, 1.0])
        )
        robot.gripper = Mock()

        for openness in (0.0, 1.0):
            robot.gripper.get_open_fraction.return_value = openness
            observation = robot.get_observations()
            self.assertEqual(observation["joint_positions"][-1], openness)
            self.assertEqual(observation["gripper_position"][0], openness)


if __name__ == "__main__":
    unittest.main()
