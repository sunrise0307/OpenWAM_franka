import unittest
from unittest.mock import patch

import numpy as np

from rot6d_deploy import main as deploy_main
from rot6d_deploy.main import execute_action_batch, wait_for_batch_end


class FakeClock:
    def __init__(self):
        self.now = 0

    def monotonic_ns(self):
        return self.now

    def sleep(self, seconds):
        self.now += round(seconds * 1e9)


class FakeRobot:
    def __init__(self, clock):
        self.clock = clock
        self.submitted = []

    def check(self):
        pass

    def apply_action(self, action):
        self.submitted.append((action, self.clock.now))
        return len(self.submitted)


class MainTests(unittest.TestCase):
    def test_group_end_has_no_extra_pause_by_default(self):
        self.assertEqual(deploy_main.build_parser().get_default("batch_pause_s"), 0)
        self.assertNotIn("--replan-lead-s", deploy_main.build_parser().format_help())

    def test_eight_targets_are_submitted_in_order_at_target_intervals(self):
        clock = FakeClock()
        robot = FakeRobot(clock)
        period_ns = round(1e9 / 30)
        next_tick, update = execute_action_batch(
            robot, list(range(8)), None, period_ns, clock.monotonic_ns, clock.sleep)
        self.assertEqual(update, 8)
        self.assertEqual([value for value, _ in robot.submitted], list(range(8)))
        self.assertEqual([timestamp for _, timestamp in robot.submitted],
                         [index * period_ns for index in range(8)])
        wait_for_batch_end(robot, next_tick, clock.monotonic_ns, clock.sleep)
        self.assertEqual(clock.now, 8 * period_ns)
        clock.now += 1_000_000_000
        execute_action_batch(robot, [8, 9], next_tick, period_ns,
                             clock.monotonic_ns, clock.sleep)
        self.assertEqual(robot.submitted[-2][1], 8 * period_ns + 1_000_000_000)
        self.assertEqual(robot.submitted[-1][1] - robot.submitted[-2][1], period_ns)

    def test_last_action_gets_a_full_period_after_a_slow_submission(self):
        clock = FakeClock()
        period_ns = round(1e9 / 30)

        class SlowRobot(FakeRobot):
            def apply_action(self, action):
                update = super().apply_action(action)
                self.clock.now += period_ns // 2
                return update

        robot = SlowRobot(clock)
        next_tick, _ = execute_action_batch(
            robot, [0, 1], None, period_ns, clock.monotonic_ns, clock.sleep)
        self.assertGreaterEqual(robot.submitted[1][1] - robot.submitted[0][1], period_ns)
        self.assertEqual(next_tick - (robot.submitted[1][1] + period_ns // 2), period_ns)
        wait_for_batch_end(robot, next_tick, clock.monotonic_ns, clock.sleep)
        self.assertEqual(clock.now, next_tick)

    def test_waits_one_second_after_the_final_action_period(self):
        clock = FakeClock()
        robot = FakeRobot(clock)
        period_ns = round(1e9 / 30)
        next_tick, _ = execute_action_batch(
            robot, [0, 1, 2], None, period_ns, clock.monotonic_ns, clock.sleep)
        pause_ns = 1_000_000_000
        wait_for_batch_end(robot, next_tick + pause_ns, clock.monotonic_ns, clock.sleep)
        self.assertEqual(clock.now, 3 * period_ns + pause_ns)

    def test_run_executes_eight_then_two_actions_from_two_replies(self):
        events = []

        class Policy:
            def __init__(self, *args, **kwargs):
                self.requests = 0
                self.next_action = 0

            def connect(self):
                return {"action_fps": 30, "action_dt_s": 1 / 30,
                        "actions_per_request": 8}

            def predict_batch(self, observation):
                self.requests += 1
                self.assert_observation(observation)
                events.append("request_%d" % self.requests)
                actions = []
                for _ in range(8):
                    self.next_action += 1
                    actions.append([0.5 + self.next_action / 1000, 0, 0.3,
                                    1, 0, 0, 0, 1, 0, 1])
                return np.asarray(actions)

            def assert_observation(self, observation):
                expected = {"type": "obs", "snapshot": self.requests,
                            "state_x": 0.5 + (2 * self.requests - 1) / 1000}
                if observation != expected:
                    raise AssertionError("unexpected observation")

            def close(self):
                pass

        class Cameras:
            def __init__(self, *args):
                self.snapshots = 0

            def start(self):
                pass

            def snapshot(self):
                self.snapshots += 1
                events.append("snapshot_%d" % self.snapshots)
                return self.snapshots, self.snapshots

            def close(self):
                pass

        class Robot:
            def __init__(self, *args):
                self.submitted = []
                self.homed = False
                self.observations = 0

            def go_home(self):
                self.homed = True
                return 0.001

            def check(self):
                pass

            def get_observation_state(self):
                if not self.homed:
                    raise AssertionError("observation requested before home motion")
                self.observations += 1
                events.append("state_%d" % self.observations)
                return [0.5 + self.observations / 1000, 0, 0.3,
                        1, 0, 0, 0, 1, 0, 1]

            def apply_action(self, action):
                self.submitted.append(float(action[0]))
                events.append("action_%d" % len(self.submitted))
                return len(self.submitted)

            def close(self):
                pass

        policy, cameras, robot = Policy(), Cameras(), Robot()
        arguments = deploy_main.build_parser().parse_args([
            "--execute", "--policy-url", "ws://example.invalid",
            "--head-serial", "head", "--wrist-serial", "wrist",
            "--prompt", "stack bowls",
            "--workspace-min", "-1", "-1", "-1",
            "--workspace-max", "1", "1", "1",
            "--max-steps", "10",
            "--batch-pause-s", "0",
        ])

        def submit_group(target_robot, actions, next_tick_ns, period_ns):
            for action in actions:
                target_robot.apply_action(action)
            return 0, len(target_robot.submitted)

        def finish_group(target_robot, deadline_ns):
            events.append("group_end_%d" % len(target_robot.submitted))

        with patch.object(deploy_main, "PolicyClient", return_value=policy), \
                patch.object(deploy_main, "DualCameras", return_value=cameras), \
                patch.object(deploy_main, "RobotController", return_value=robot), \
                patch.object(deploy_main, "build_observation",
                             side_effect=lambda head, wrist, state, prompt:
                             {"type": "obs", "snapshot": head, "state_x": state[0]}), \
                patch.object(deploy_main, "execute_action_batch", side_effect=submit_group), \
                patch.object(deploy_main, "wait_for_batch_end", side_effect=finish_group):
            deploy_main.run(arguments)

        self.assertEqual(policy.requests, 2)
        self.assertEqual(cameras.snapshots, 2)
        self.assertEqual(robot.observations, 4)
        self.assertTrue(robot.homed)
        self.assertLess(events.index("action_8"), events.index("group_end_8"))
        self.assertLess(events.index("group_end_8"), events.index("snapshot_2"))
        self.assertLess(events.index("snapshot_2"), events.index("state_3"))
        self.assertLess(events.index("state_3"), events.index("request_2"))
        np.testing.assert_allclose(robot.submitted,
                                   [0.5 + index / 1000 for index in range(1, 11)])

    def test_failed_home_motion_stops_before_policy_request(self):
        events = []

        class Policy:
            def connect(self):
                return {"action_fps": 30, "action_dt_s": 1 / 30,
                        "actions_per_request": 8}

            def predict_batch(self, observation):
                events.append("predict")
                raise AssertionError("policy request must not run")

            def close(self):
                events.append("policy_closed")

        class Cameras:
            def __init__(self, *args):
                pass

            def start(self):
                pass

            def snapshot(self):
                events.append("snapshot")
                raise AssertionError("camera observation must not run")

            def close(self):
                events.append("cameras_closed")

        class Robot:
            def __init__(self, *args):
                pass

            def go_home(self):
                events.append("home")
                raise RuntimeError("home motion failed")

            def close(self):
                events.append("robot_closed")

        arguments = deploy_main.build_parser().parse_args([
            "--execute", "--policy-url", "ws://example.invalid",
            "--head-serial", "head", "--wrist-serial", "wrist",
            "--prompt", "stack bowls",
            "--workspace-min", "-1", "-1", "-1",
            "--workspace-max", "1", "1", "1",
        ])
        with patch.object(deploy_main, "PolicyClient", return_value=Policy()), \
                patch.object(deploy_main, "DualCameras", return_value=Cameras()), \
                patch.object(deploy_main, "RobotController", return_value=Robot()):
            with self.assertRaisesRegex(RuntimeError, "home motion failed"):
                deploy_main.run(arguments)

        self.assertEqual(events, ["home", "robot_closed", "cameras_closed", "policy_closed"])


if __name__ == "__main__":
    unittest.main()
