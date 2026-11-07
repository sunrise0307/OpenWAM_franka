"""A continuously running Cartesian controller for validated absolute targets."""

from dataclasses import dataclass
import math
import threading
import time

import numpy as np

from .geometry import (
    RECORDED_GRIPPER_MAX_WIDTH_M,
    gripper_fraction_to_width,
    gripper_width_to_fraction,
    measured_state,
    parse_action,
    quaternion_distance,
    unit_quaternion,
)


@dataclass(frozen=True)
class SafetyLimits:
    workspace_min: tuple
    workspace_max: tuple
    maximum_target_distance_m: float = 0.12
    maximum_target_rotation_rad: float = 0.8
    max_linear_velocity_m_s: float = 0.8
    max_angular_velocity_rad_s: float = 2.5
    target_timeout_s: float = 5.0

    def __post_init__(self):
        lower = np.asarray(self.workspace_min, dtype=np.float64)
        upper = np.asarray(self.workspace_max, dtype=np.float64)
        if lower.shape != (3,) or upper.shape != (3,) or not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("workspace bounds must each contain three finite numbers")
        if np.any(lower >= upper):
            raise ValueError("each workspace minimum must be below its maximum")
        for name in ("maximum_target_distance_m", "maximum_target_rotation_rad",
                     "max_linear_velocity_m_s", "max_angular_velocity_rad_s", "target_timeout_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive and finite" % name)
        if self.target_timeout_s < 0.1:
            raise ValueError("target timeout must be at least 0.1 seconds")


class RobotController:
    """Own the sole Polymetis arm controller for one deployment session."""

    def __init__(self, host, arm_port, gripper_port, limits, gripper_speed=0.05,
                 gripper_force=20.0):
        from polymetis import GripperInterface, RobotInterface

        self.arm = RobotInterface(ip_address=host, port=arm_port)
        try:
            self.gripper = GripperInterface(ip_address=host, port=gripper_port)
        except Exception:
            self.arm.channel.close()
            raise
        self.limits = limits
        self.gripper_speed = gripper_speed
        self.gripper_force = gripper_force
        try:
            metadata = self.arm.metadata
            if metadata.dof != 7 or metadata.ee_link_name != "fr3_link8":
                raise RuntimeError("robot metadata must describe a seven-joint FR3 with fr3_link8")
            if metadata.hz != 1000:
                raise RuntimeError("expected the FR3 hardware controller to run at 1000 hertz")
            if (not hasattr(self.gripper, "metadata") or
                    self.gripper.metadata.max_width < RECORDED_GRIPPER_MAX_WIDTH_M):
                raise RuntimeError("gripper cannot reach the recorded maximum width")
            if (not math.isfinite(gripper_speed) or gripper_speed <= 0 or
                    not math.isfinite(gripper_force) or gripper_force <= 0):
                raise ValueError("gripper speed and force must be positive")
            if self.arm.is_running_policy():
                raise RuntimeError("another robot controller is already running")
        except Exception:
            self.arm.channel.close()
            self.gripper.channel.close()
            raise

        self._lock = threading.RLock()
        self._gripper_condition = threading.Condition()
        self._stop = threading.Event()
        self._active = False
        self._closed = False
        self._fault = None
        self._last_target_ns = None
        self._pending_gripper = None
        self._last_gripper_target = None
        self._sequence = 0
        self._watchdog = threading.Thread(target=self._watchdog_loop, name="target_watchdog", daemon=True)
        self._gripper_thread = threading.Thread(target=self._gripper_loop, name="gripper_command", daemon=True)
        self._watchdog.start()
        self._gripper_thread.start()

    def _read_pose(self):
        import torch

        state = self.arm.get_robot_state()
        joints = torch.as_tensor(state.joint_positions, dtype=torch.float32)
        if joints.shape != (7,) or not torch.isfinite(joints).all():
            raise RuntimeError("robot returned invalid joint positions")
        position, quaternion = self.arm.robot_model.forward_kinematics(joints)
        position = position.detach().cpu().numpy().astype(np.float64)
        quaternion = unit_quaternion(quaternion.detach().cpu().numpy())
        if not np.isfinite(position).all():
            raise RuntimeError("robot returned invalid end-effector position")
        return position, quaternion

    def get_observation_state(self):
        self.check()
        position, quaternion = self._read_pose()
        openness = gripper_width_to_fraction(self.gripper.get_state().width)
        return measured_state(position, quaternion, openness)

    def _inside_workspace(self, position):
        lower = np.asarray(self.limits.workspace_min)
        upper = np.asarray(self.limits.workspace_max)
        return bool(np.all(position >= lower) and np.all(position <= upper))

    def go_home(self):
        """Finish the robot's configured home motion before accepting policy targets."""
        with self._lock:
            self.check()
            if self._active or self.arm.is_running_policy():
                raise RuntimeError("another robot controller is already running")

            home_joints = self.arm.home_pose.detach().cpu().numpy()
            home_position, _ = self.arm.robot_model.forward_kinematics(self.arm.home_pose)
            home_position = home_position.detach().cpu().numpy()
            if not self._inside_workspace(home_position):
                raise RuntimeError("configured workspace excludes the robot home pose")

            try:
                self.arm.go_home(timeout=30.0)
            except BaseException:
                if self.arm.is_running_policy():
                    self.arm.terminate_current_policy(return_log=False)
                raise

            if self.arm.is_running_policy():
                self.arm.terminate_current_policy(return_log=False)
                raise RuntimeError("robot home motion did not finish")
            state = self.arm.get_robot_state()
            if state.error_code != 0:
                raise RuntimeError("robot reported an error during home motion: %s" % state.error_code)
            actual_joints = np.asarray(state.joint_positions, dtype=np.float64)
            if actual_joints.shape != (7,) or not np.isfinite(actual_joints).all():
                raise RuntimeError("robot returned invalid joint positions after home motion")
            maximum_error = float(np.max(np.abs(actual_joints - home_joints)))
            if maximum_error > 0.05:
                raise RuntimeError("robot did not reach home pose; maximum joint error %.3f rad" % maximum_error)
            return maximum_error

    def _start_locked(self):
        import torch
        from .cartesian_policy import WatchdogCartesianImpedanceControl

        if self.arm.is_running_policy():
            raise RuntimeError("another robot controller started during deployment setup")
        policy = WatchdogCartesianImpedanceControl(
            joint_pos_current=self.arm.get_joint_positions(),
            Kp=torch.as_tensor(self.arm.metadata.default_Kx, dtype=torch.float32),
            Kd=torch.as_tensor(self.arm.metadata.default_Kxd, dtype=torch.float32),
            robot_model=self.arm.robot_model,
            watchdog_ticks=round(self.limits.target_timeout_s * self.arm.metadata.hz),
            control_hz=self.arm.metadata.hz,
            max_linear_velocity_m_s=self.limits.max_linear_velocity_m_s,
            max_angular_velocity_rad_s=self.limits.max_angular_velocity_rad_s,
            ignore_gravity=self.arm.use_grav_comp,
        )
        self.arm.send_torch_policy(policy, blocking=False)
        self._active = True

    def _terminate_locked(self):
        if self._active:
            try:
                self.arm.terminate_current_policy(return_log=False)
            finally:
                self._active = False
                self._last_target_ns = None

    def apply_action(self, raw_action):
        """Validate and send the full absolute target to the running controller."""
        import torch

        target = parse_action(raw_action)
        measured_position, measured_quaternion = self._read_pose()
        if not self._inside_workspace(measured_position):
            raise RuntimeError("measured end-effector pose is outside the configured workspace")
        if not self._inside_workspace(target.position):
            raise RuntimeError("model target is outside the configured workspace")
        if np.linalg.norm(target.position - measured_position) > self.limits.maximum_target_distance_m:
            raise RuntimeError("model target is too far from the measured position")
        if quaternion_distance(target.quaternion_xyzw, measured_quaternion) > self.limits.maximum_target_rotation_rad:
            raise RuntimeError("model target is too far from the measured orientation")

        with self._lock:
            self.check()
            if not self._active:
                self._start_locked()
            try:
                self._sequence += 1
                update_index = self.arm.update_current_policy({
                    "ee_pos_desired": torch.as_tensor(target.position, dtype=torch.float32),
                    "ee_quat_desired": torch.as_tensor(target.quaternion_xyzw, dtype=torch.float32),
                    "heartbeat": torch.tensor([float(self._sequence)], dtype=torch.float32),
                })
                if update_index < 0:
                    raise RuntimeError("robot rejected the Cartesian target update")
            except Exception as exc:
                self._fault = "Cartesian target update failed: %s" % exc
                self._terminate_locked()
                raise
            self._last_target_ns = time.monotonic_ns()

        with self._gripper_condition:
            if (self._last_gripper_target is None or
                    abs(target.gripper_open_fraction - self._last_gripper_target) >= 0.02):
                self._pending_gripper = target.gripper_open_fraction
                self._gripper_condition.notify()
        return update_index

    def _watchdog_loop(self):
        while not self._stop.wait(0.05):
            with self._lock:
                if not self._active or self._last_target_ns is None:
                    continue
                expired = (time.monotonic_ns() - self._last_target_ns >
                           self.limits.target_timeout_s * 1e9)
                if not expired and self._fault is None:
                    continue
                if self._fault is None:
                    self._fault = "policy target timed out"
                try:
                    self._terminate_locked()
                except Exception as exc:
                    self._fault += "; controller termination failed: %s" % exc

    def _gripper_loop(self):
        last_command_at = 0.0
        while not self._stop.is_set():
            with self._gripper_condition:
                while self._pending_gripper is None and not self._stop.is_set():
                    self._gripper_condition.wait(timeout=0.1)
                if self._stop.is_set():
                    return
                value = self._pending_gripper
                self._pending_gripper = None
            remaining = max(0.0, 0.05 - (time.monotonic() - last_command_at))
            if self._stop.wait(remaining):
                return
            with self._gripper_condition:
                if self._pending_gripper is not None:
                    value = self._pending_gripper
                    self._pending_gripper = None
            try:
                self.gripper.goto(
                    width=gripper_fraction_to_width(value),
                    speed=self.gripper_speed, force=self.gripper_force, blocking=True
                )
            except Exception as exc:
                with self._lock:
                    self._fault = "gripper command failed: %s" % exc
                return
            last_command_at = time.monotonic()
            with self._gripper_condition:
                self._last_gripper_target = value

    def check(self):
        if self._closed:
            raise RuntimeError("robot controller is closed")
        if self._fault is not None:
            raise RuntimeError(self._fault)

    def close(self):
        self._stop.set()
        with self._gripper_condition:
            self._gripper_condition.notify_all()
        self._watchdog.join(timeout=2)
        self._gripper_thread.join(timeout=2)
        with self._lock:
            if not self._closed:
                try:
                    self._terminate_locked()
                finally:
                    self._closed = True
                    self.arm.channel.close()
                    self.gripper.channel.close()
