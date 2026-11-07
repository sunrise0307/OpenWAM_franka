import unittest
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import torch

from teleop.robots.eef import (
    close_fraction_to_width,
    eef10,
    limit_rotation_step,
    pose_from_column_major_transform,
    rotation_matrix_from_xyzw,
    width_to_close_fraction,
    xyzw_from_matrix,
)


class EefConventionsTest(unittest.TestCase):
    def test_joint_step_allowance_keeps_hard_limits(self):
        from teleop.robots.fr3 import fr3Robot

        robot = fr3Robot.__new__(fr3Robot)
        robot.eef_mode = True
        robot._watchdog_tripped = False
        robot._command_seq = 0
        robot._max_rotation_step = 0.10
        robot._max_joint_step = np.array([0.30] * 7)
        robot._gripper_closed_width = 0.0
        robot._gripper_open_width = 0.08
        robot._last_gripper_target_width = 0.08
        robot._workspace_lower = None
        robot._workspace_upper = None
        robot._state_and_pose = lambda: (None, np.zeros(7), None,
                                          np.zeros(3), np.eye(3))
        robot.robot = Mock()
        robot.robot.robot_model.get_joint_angle_limits.return_value = (
            torch.full((7,), -3.0), torch.full((7,), 3.0))
        robot.robot.robot_model.forward_kinematics.return_value = (
            torch.tensor([0.12, 0.0, 0.0]), torch.tensor([0.0, 0.0, 0.0, 1.0]))
        robot.robot.update_desired_joint_positions.return_value = 1
        robot.gripper = Mock()

        wrist_target = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.27, 0.0])
        robot.robot.solve_inverse_kinematics.return_value = (wrist_target, True)
        accepted = robot.command_joint_state(np.zeros(8))
        self.assertEqual(accepted["controller_update_idx"], 1)
        np.testing.assert_allclose(accepted["target_joint_pos"], wrist_target)
        np.testing.assert_allclose(robot.robot.update_desired_joint_positions.call_args.args[0],
                                   wrist_target)
        robot.robot.update_desired_ee_pose.assert_not_called()
        self.assertIsNone(accepted["gripper_command_ack_monotonic_ns"])
        np.testing.assert_allclose(
            robot.robot.solve_inverse_kinematics.call_args.args[0], [0.12, 0.0, 0.0],
            atol=1e-7)

        wrist_over_limit = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.31, 0.0])
        robot.robot.solve_inverse_kinematics.return_value = (wrist_over_limit, True)
        with self.assertRaisesRegex(ValueError, "J6 step 0.310 rad exceeds 0.300"):
            robot.command_joint_state(np.zeros(8))

        arm_target = torch.tensor([0.27, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        robot.robot.solve_inverse_kinematics.return_value = (arm_target, True)
        self.assertEqual(robot.command_joint_state(np.zeros(8))["controller_update_idx"], 1)

        arm_over_limit = torch.tensor([0.31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        robot.robot.solve_inverse_kinematics.return_value = (arm_over_limit, True)
        with self.assertRaisesRegex(ValueError, "J1 step 0.310 rad exceeds 0.300"):
            robot.command_joint_state(np.zeros(8))

        robot.robot.robot_model.get_joint_angle_limits.return_value = (
            torch.full((7,), -0.1), torch.full((7,), 0.1))
        robot.robot.solve_inverse_kinematics.return_value = (wrist_target, True)
        with self.assertRaisesRegex(ValueError, "IK target outside FR3 joint limits"):
            robot.command_joint_state(np.zeros(8))

    def test_franka_ee_pose_is_converted_to_model_link8(self):
        from teleop.robots.fr3 import fr3Robot

        robot = fr3Robot.__new__(fr3Robot)
        robot.eef_mode = True
        robot.robot = Mock()
        robot.robot.get_robot_state.return_value = Mock(
            joint_positions=[0.0] * 7, joint_velocities=[0.0] * 7,
            measured_ee_transform=np.array([
                [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.2],
                [-np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.3],
                [0.0, 0.0, 1.0, 0.6034],
                [0.0, 0.0, 0.0, 1.0],
            ]).reshape(16, order="F"),
            callback_monotonic_ns=1,
        )
        robot.robot.robot_model.forward_kinematics.return_value = (
            torch.tensor([0.2, 0.3, 0.5]), torch.tensor([0.0, 0.0, 0.0, 1.0]))

        _, _, _, position, rotation = robot._state_and_pose()
        np.testing.assert_allclose(position, [0.2, 0.3, 0.5], atol=1e-7)
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-7)
        robot._validate_model_against_measured_pose()

        robot.robot.get_robot_state.return_value.measured_ee_transform[14] += 0.03
        with self.assertRaisesRegex(RuntimeError, "FK/O_T_EE mismatch"):
            robot._validate_model_against_measured_pose()

    def test_direct_eef_pose_uses_one_inverse_kinematics_solve(self):
        from teleop.robots.fr3 import fr3Robot

        robot = fr3Robot.__new__(fr3Robot)
        robot.eef_mode = True
        robot._watchdog_tripped = False
        robot._command_seq = 0
        robot._max_rotation_step = 0.10
        robot._max_joint_step = np.array([0.30] * 7)
        robot._gripper_closed_width = 0.0
        robot._gripper_open_width = 0.08
        robot._last_gripper_target_width = 0.04
        robot._workspace_lower = None
        robot._workspace_upper = None
        robot._state_and_pose = lambda: (None, np.zeros(7), None,
                                          np.zeros(3), np.eye(3))
        robot.robot = Mock()
        robot.robot.robot_model.get_joint_angle_limits.return_value = (
            torch.full((7,), -1.0), torch.full((7,), 1.0))
        robot.robot.solve_inverse_kinematics.return_value = (
            torch.tensor([0.1] + [0.0] * 6), True)
        robot.robot.update_desired_joint_positions.return_value = 1
        robot.gripper = Mock()

        accepted = robot.command_eef_pose([0.12, 0.0, 0.4], [0.0, 0.0, 0.0, 1.0], 0.5)
        robot.robot.solve_inverse_kinematics.assert_called_once()
        robot.robot.robot_model.forward_kinematics.assert_not_called()
        np.testing.assert_allclose(accepted["target_eef10"],
                                   [0.12, 0.0, 0.4, 1, 0, 0, 0, 1, 0, 0])
        np.testing.assert_allclose(accepted["target_joint_pos"], [0.1] + [0.0] * 6)

        robot._workspace_lower = np.array([0.2, -1.0, -1.0])
        robot._workspace_upper = np.array([1.0, 1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "outside configured workspace"):
            robot.command_eef_pose([0.12, 0.0, 0.4], [0.0, 0.0, 0.0, 1.0], 0.5)
        robot.robot.solve_inverse_kinematics.assert_called_once()

    def test_eef_mode_requires_explicit_model_before_connecting(self):
        from teleop.robots.fr3 import fr3Robot

        with self.assertRaisesRegex(ValueError, "requires --fr3-urdf"):
            fr3Robot(eef_mode=True)

    def test_workspace_bounds_are_optional_but_enforced_when_present(self):
        from teleop.robots.fr3 import fr3Robot

        robot = fr3Robot.__new__(fr3Robot)
        robot.eef_mode = True
        robot._watchdog_tripped = False
        robot._command_seq = 0
        robot._max_rotation_step = 0.10
        robot._max_joint_step = 0.12
        robot._gripper_closed_width = 0.0
        robot._gripper_open_width = 0.08
        robot._last_gripper_target_width = 0.08
        robot._workspace_lower = None
        robot._workspace_upper = None
        robot._state_and_pose = lambda: (None, np.zeros(7), None,
                                          np.zeros(3), np.eye(3))
        robot.robot = Mock()
        robot.robot.robot_model.get_joint_angle_limits.return_value = (
            torch.full((7,), -1.0), torch.full((7,), 1.0))
        robot.robot.robot_model.forward_kinematics.return_value = (
            torch.tensor([0.01, 0.0, 0.0]), torch.tensor([0.0, 0.0, 0.0, 1.0]))
        robot.robot.solve_inverse_kinematics.return_value = (torch.zeros(7), True)
        robot.robot.update_desired_joint_positions.return_value = 1
        robot.gripper = Mock()

        self.assertEqual(robot.command_joint_state(np.zeros(8))["controller_update_idx"], 1)
        robot._workspace_lower = np.array([0.02, -1.0, -1.0])
        robot._workspace_upper = np.array([1.0, 1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "outside configured workspace"):
            robot.command_joint_state(np.zeros(8))

    def test_one_workspace_bound_is_rejected_before_connecting(self):
        from teleop.robots.fr3 import fr3Robot

        with tempfile.NamedTemporaryFile() as urdf, patch("polymetis.RobotInterface") as interface:
            with self.assertRaisesRegex(ValueError, "both workspace bounds or neither"):
                fr3Robot(eef_mode=True, fr3_urdf=urdf.name, ee_link_name="fr3_link8",
                         workspace_lower=(0.0, 0.0, 0.0))
            interface.assert_not_called()

    def test_column_major_pose_and_rot6d(self):
        transform = np.eye(4)
        transform[:3, 3] = [0.3, -0.2, 0.5]
        position, rotation = pose_from_column_major_transform(
            transform.reshape(16, order="F"))
        np.testing.assert_allclose(position, [0.3, -0.2, 0.5])
        np.testing.assert_allclose(eef10(position, rotation, 0.08, 0, 0.08),
                                   [0.3, -0.2, 0.5, 1, 0, 0, 0, 1, 0, 1])

    def test_gripper_direction(self):
        self.assertEqual(width_to_close_fraction(0.08, 0, 0.08), 0)
        self.assertEqual(width_to_close_fraction(0, 0, 0.08), 1)
        self.assertAlmostEqual(close_fraction_to_width(0.5, 0, 0.08), 0.04)

    def test_release_waits_until_target_exceeds_grasped_width(self):
        from teleop.robots.fr3 import fr3Robot

        robot = fr3Robot.__new__(fr3Robot)
        robot.eef_mode = False
        robot._watchdog_tripped = False
        robot._command_seq = 0
        robot._gripper_closed_width = 0.0
        robot._gripper_open_width = 0.08
        robot._gripper_speed = 0.05
        robot._gripper_force = 20.0
        robot._last_gripper_target_width = 0.0
        robot.robot = Mock()
        robot.gripper = Mock()
        robot.gripper.get_state.return_value.width = 0.03

        robot.command_joint_state(np.r_[np.zeros(7), 0.975])
        robot.gripper.grasp.assert_not_called()
        robot.gripper.goto.assert_not_called()

        robot.command_joint_state(np.r_[np.zeros(7), 0.55])
        robot.gripper.goto.assert_called_once()
        self.assertAlmostEqual(robot.gripper.goto.call_args.kwargs["width"], 0.036)
        self.assertEqual(robot.gripper.goto.call_args.kwargs["blocking"], True)
        robot.gripper.grasp.assert_not_called()

        robot.command_joint_state(np.r_[np.zeros(7), 1.0])
        robot.gripper.grasp.assert_called_once()

    def test_quaternion_round_trip_and_zero_rejection(self):
        for quaternion in ([0.3, -0.2, 0.1, 0.9], [1, 0, 0, 0],
                           [0, 1, 0, 0], [0, 0, 1, 0]):
            matrix = rotation_matrix_from_xyzw(quaternion)
            np.testing.assert_allclose(rotation_matrix_from_xyzw(xyzw_from_matrix(matrix)),
                                       matrix, atol=1e-12)
        with self.assertRaises(ValueError):
            limit_rotation_step(np.zeros(4), [0, 0, 0, 1], 0.1)


if __name__ == "__main__":
    unittest.main()
