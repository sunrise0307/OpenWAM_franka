"""Run one policy session against one FR3 robot."""

import argparse
from contextlib import ExitStack
import logging
import math
import os
import time

import numpy as np

from .control.geometry import parse_action, quaternion_distance, rotation_six_to_quaternion_xyzw
from .control.robot import RobotController, SafetyLimits
from .policy.cameras import DualCameras
from .policy.client import PolicyClient, build_observation


LOGGER = logging.getLogger("rot6d_deploy")
TARGET_FREQUENCY = 30.0


def execute_action_batch(robot, actions, next_tick_ns, period_ns, clock_ns=None, sleeper=None):
    """Submit targets in order, spacing them from actual submission times."""
    clock_ns = time.monotonic_ns if clock_ns is None else clock_ns
    sleeper = time.sleep if sleeper is None else sleeper
    update_index = None
    for action in actions:
        robot.check()
        now_ns = clock_ns()
        deadline_ns = now_ns if next_tick_ns is None else max(now_ns, next_tick_ns)
        if deadline_ns > now_ns:
            sleeper((deadline_ns - now_ns) / 1e9)
        update_index = robot.apply_action(action)
        next_tick_ns = max(deadline_ns, clock_ns()) + period_ns
    return next_tick_ns, update_index


def wait_for_batch_end(robot, deadline_ns, clock_ns=None, sleeper=None):
    """Wait through the final action period and any requested pause."""
    clock_ns = time.monotonic_ns if clock_ns is None else clock_ns
    sleeper = time.sleep if sleeper is None else sleeper
    while True:
        robot.check()
        remaining_ns = deadline_ns - clock_ns()
        if remaining_ns <= 0:
            return
        sleeper(min(remaining_ns / 1e9, 0.1))


def fetch_policy_actions(policy, head_image, wrist_image, state, prompt):
    request = build_observation(head_image, wrist_image, state, prompt)
    started_ns = time.monotonic_ns()
    actions = policy.predict_batch(request)
    return actions, (time.monotonic_ns() - started_ns) / 1e9


def build_parser():
    parser = argparse.ArgumentParser(description="Track absolute ten-value policy targets on FR3")
    parser.add_argument("--execute", action="store_true", help="Required before connecting to the real robot")
    parser.add_argument("--policy-url", required=True)
    parser.add_argument("--robot-host", default="127.0.0.1")
    parser.add_argument("--arm-port", type=int, default=50051)
    parser.add_argument("--gripper-port", type=int, default=50052)
    parser.add_argument("--head-serial", required=True)
    parser.add_argument("--wrist-serial", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--workspace-min", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--workspace-max", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--policy-timeout", type=float, default=30.0)
    parser.add_argument("--target-timeout", type=float, default=5.0)
    parser.add_argument("--batch-pause-s", type=float, default=0,
                        help="Optional delay after the final action period before collecting a new observation")
    parser.add_argument("--max-linear-velocity", type=float, default=0.8,
                        help="Maximum reference position speed in meters per second")
    parser.add_argument("--max-angular-velocity", type=float, default=2.5,
                        help="Maximum reference rotation speed in radians per second")
    parser.add_argument("--max-steps", type=int)
    return parser


def run(arguments):
    if not arguments.execute:
        raise ValueError("pass --execute to connect and command the robot")
    if arguments.max_steps is not None and arguments.max_steps <= 0:
        raise ValueError("maximum step count must be positive")
    if not math.isfinite(arguments.batch_pause_s) or arguments.batch_pause_s < 0:
        raise ValueError("batch pause must be finite and nonnegative")
    limits = SafetyLimits(
        workspace_min=tuple(arguments.workspace_min),
        workspace_max=tuple(arguments.workspace_max),
        target_timeout_s=arguments.target_timeout,
        max_linear_velocity_m_s=arguments.max_linear_velocity,
        max_angular_velocity_rad_s=arguments.max_angular_velocity,
    )
    with ExitStack() as cleanup:
        policy = PolicyClient(arguments.policy_url, expected_frequency=TARGET_FREQUENCY,
                              timeout_s=arguments.policy_timeout)
        cleanup.callback(policy.close)
        metadata = policy.connect()
        LOGGER.info("Policy contract accepted: %d actions per response, %.1f target updates per second",
                    metadata["actions_per_request"], metadata["action_fps"])
        cameras = DualCameras(arguments.head_serial, arguments.wrist_serial)
        cleanup.callback(cameras.close)
        cameras.start()

        robot = RobotController(arguments.robot_host, arguments.arm_port,
                                arguments.gripper_port, limits)
        cleanup.callback(robot.close)
        LOGGER.info("Returning robot to its configured home pose")
        home_error = robot.go_home()
        LOGGER.info("Robot home motion complete; maximum joint error %.4f rad", home_error)
        period_ns = round(1e9 * metadata["action_dt_s"])
        pause_ns = round(1e9 * arguments.batch_pause_s)
        next_tick_ns = None
        step = 0
        while arguments.max_steps is None or step < arguments.max_steps:
            robot.check()
            head_image, wrist_image = cameras.snapshot()
            state = robot.get_observation_state()
            actions, fetch_seconds = fetch_policy_actions(
                policy, head_image, wrist_image, state, arguments.prompt)
            if arguments.max_steps is not None:
                actions = actions[:arguments.max_steps - step]
            count = len(actions)
            robot.check()
            measured_position = np.asarray(state[:3], dtype=np.float64)
            measured_quaternion = rotation_six_to_quaternion_xyzw(state[3:9])
            targets = [parse_action(action) for action in actions]
            largest_position_offset = max(float(np.linalg.norm(target.position - measured_position))
                                          for target in targets)
            largest_rotation_offset = max(quaternion_distance(
                target.quaternion_xyzw, measured_quaternion) for target in targets)
            next_tick_ns, update_index = execute_action_batch(
                robot, actions, next_tick_ns, period_ns)
            wait_for_batch_end(robot, next_tick_ns + pause_ns)
            state_after = robot.get_observation_state()
            measured_position_after = np.asarray(state_after[:3], dtype=np.float64)
            measured_quaternion_after = rotation_six_to_quaternion_xyzw(state_after[3:9])
            measured_displacement = float(np.linalg.norm(measured_position_after - measured_position))
            measured_rotation = quaternion_distance(measured_quaternion_after, measured_quaternion)
            final_position_error = float(np.linalg.norm(targets[-1].position - measured_position_after))
            final_rotation_error = quaternion_distance(
                targets[-1].quaternion_xyzw, measured_quaternion_after)
            step += count
            LOGGER.info(
                "Policy steps %d-%d, robot update %d, fetch %.3f s, largest target offset %.4f m / %.3f rad; "
                "measured movement %.4f m / %.3f rad; final target error %.4f m / %.3f rad",
                step - count + 1, step, update_index, fetch_seconds,
                largest_position_offset, largest_rotation_offset,
                measured_displacement, measured_rotation,
                final_position_error, final_rotation_error,
            )
            target_positions = np.asarray([target.position for target in targets])
            target_span = float(np.max(np.linalg.norm(
                target_positions - target_positions[0], axis=1)))
            LOGGER.info(
                "Policy steps %d-%d positions: observed=%s first_target=%s last_target=%s "
                "target_span=%.5f m measured_after=%s",
                step - count + 1, step,
                np.array2string(measured_position, precision=5),
                np.array2string(target_positions[0], precision=5),
                np.array2string(target_positions[-1], precision=5),
                target_span,
                np.array2string(measured_position_after, precision=5),
            )


def main():
    parser = build_parser()
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if arguments.execute:
            from .runtime import bootstrap, preflight_imports

            bootstrap()
            preflight_imports()
            LOGGER.info("Polymetis libraries verified from %s", os.environ["POLYMETIS_BUILD_DIR"])
        run(arguments)
    except KeyboardInterrupt:
        LOGGER.info("Deployment interrupted")
    except Exception:
        LOGGER.exception("Deployment stopped")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
