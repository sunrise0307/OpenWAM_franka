"""Offline checks for smooth direct teleoperation commands."""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from experiments.run_env import (
    TeleopCommandSmoother, check_startup_alignment, record_teleop_command)
from teleop.env import RobotEnv


class DirectTeleopControlTest(unittest.TestCase):
    def test_joint_jump_is_spread_across_commands(self):
        smoother = TeleopCommandSmoother()
        smoother.reset(np.zeros(7), 1_000_000_000)
        previous = np.zeros(7)
        for tick in range(1, 11):
            command = smoother.step(np.r_[np.full(7, 0.3), 0.7],
                                    1_000_000_000 + tick * 10_000_000)
            self.assertTrue(np.all(command[:7] > previous))
            self.assertTrue(np.all(command[:7] - previous <= 0.015 + 1e-12))
            self.assertEqual(command[7], 0.7)
            previous = command[:7]
        self.assertTrue(np.all(previous < 0.3))

    def test_command_only_cycle_does_not_read_robot_feedback(self):
        robot = Mock()
        robot.num_dofs.return_value = 8
        env = RobotEnv(robot, control_rate_hz=100)
        env._rate.sleep = Mock()
        target = np.zeros(8)
        env.command(target)
        env.command(target)
        self.assertEqual(robot.command_joint_state.call_count, 2)
        robot.get_observations.assert_not_called()
        robot.num_dofs.assert_called_once()

    def test_recording_pairs_observation_before_command(self):
        events = []
        env = Mock()
        observation = {"joint_positions": np.zeros(8)}
        env.get_obs.side_effect = lambda: events.append("observe") or observation
        env.command.side_effect = lambda _action: events.append("command")
        action = np.ones(8)
        with patch("experiments.run_env.save_frame") as save:
            save.side_effect = lambda *_args: events.append("save")
            result = record_teleop_command(env, action, "recording", "time")
            save.assert_called_once_with("recording", "time", observation, action)
        self.assertIs(result, observation)
        self.assertEqual(events, ["observe", "command", "save"])

    def test_startup_alignment_checks_both_directions_and_ignores_gripper(self):
        check_startup_alignment(np.r_[np.zeros(7), 1.0], np.zeros(8))
        with self.assertRaises(ValueError):
            check_startup_alignment(np.r_[[-0.6] + [0.0] * 6, 0.0], np.zeros(8))


if __name__ == "__main__":
    unittest.main()
