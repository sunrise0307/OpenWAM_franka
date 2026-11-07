"""The measured robot and submitted command record for one control cycle."""

import queue
import threading
import time
from pathlib import Path

import numpy as np


FIELDS = {
    "timestamps/control_cycle_id": ((), "i8", -1),
    "timestamps/observation_ns": ((), "i8", -1),
    "timestamps/command_ns": ((), "i8", -1),
    "timestamps/gripper_observation_ns": ((), "i8", -1),
    "timestamps/gripper_command_ns": ((), "i8", -1),
    "timestamps/arm_device_sample_time_ms": ((), "i8", -1),
    "timestamps/gripper_device_sample_time_ms": ((), "i8", -1),
    "observations/joint_positions": ((7,), "f8", np.nan),
    "observations/joint_velocities": ((7,), "f8", np.nan),
    "observations/eef_pose": ((7,), "f8", np.nan),
    "observations/eef_pose_6d": ((9,), "f8", np.nan),
    "observations/gripper_state": ((1,), "f8", np.nan),
    "actions/target_joint_positions": ((7,), "f8", np.nan),
    "actions/target_eef_pose": ((7,), "f8", np.nan),
    "actions/target_eef_pose_6d": ((9,), "f8", np.nan),
    "actions/target_gripper": ((1,), "f8", np.nan),
    "diagnostics/observation_valid": ((), "?", False),
    "diagnostics/command_valid": ((), "?", False),
    "diagnostics/command_status": ((), "u1", 0),
    "diagnostics/controller_update_idx": ((), "i8", -1),
    "diagnostics/gripper_command_status": ((), "u1", 0),
    "diagnostics/observation_error": ((), "S256", b""),
    "diagnostics/command_error": ((), "S256", b""),
}


class Fr3Kinematics:
    """Use one robot model and one tool frame for observed and target joints."""

    def __init__(self, urdf, frame="fr3_link8"):
        import pinocchio as pin

        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId(frame)
        if self.model.nq != 7 or self.frame_id >= self.model.nframes:
            raise ValueError("Robot model must contain seven joints and the selected tool frame")
        self.lower = np.asarray(self.model.lowerPositionLimit, dtype=np.float64)
        self.upper = np.asarray(self.model.upperPositionLimit, dtype=np.float64)

    def pose(self, joints):
        joints = np.asarray(joints, dtype=np.float64)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError("Seven finite joint positions are required")
        pin = self.pin
        pin.forwardKinematics(self.model, self.data, joints)
        pin.updateFramePlacements(self.model, self.data)
        transform = self.data.oMf[self.frame_id]
        position = np.asarray(transform.translation, dtype=np.float64)
        rotation = np.asarray(transform.rotation, dtype=np.float64)
        quaternion = np.asarray(pin.Quaternion(rotation).coeffs(), dtype=np.float64)
        rotation_6d = rotation[:, :2].T.reshape(6)
        return np.r_[position, quaternion], np.r_[position, rotation_6d]

    def within_limits(self, joints, tolerance=1e-8):
        joints = np.asarray(joints, dtype=np.float64)
        return (joints.shape == (7,) and np.isfinite(joints).all() and
                np.all(joints >= self.lower - tolerance) and
                np.all(joints <= self.upper + tolerance))


class JointTargetSmoother:
    """Follow sampled joint targets smoothly at an irregular command rate."""

    def __init__(self, time_constant_s=0.03, max_speed_rad_s=1.5):
        if (not np.isfinite(time_constant_s) or time_constant_s <= 0 or
                not np.isfinite(max_speed_rad_s) or max_speed_rad_s <= 0):
            raise ValueError("Joint smoothing settings must be positive and finite")
        self.time_constant_s = float(time_constant_s)
        self.max_speed_rad_s = float(max_speed_rad_s)
        self._input = None
        self._command = None
        self._last_ns = None

    def reset(self, joints, now_ns):
        joints = np.asarray(joints, dtype=np.float64)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError("Seven finite starting joint positions are required")
        self._input = joints.copy()
        self._command = joints.copy()
        self._last_ns = int(now_ns)

    def step(self, target, now_ns):
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError("Seven finite target joint positions are required")
        if self._last_ns is None:
            raise RuntimeError("Joint smoother must start from an acknowledged target")
        now_ns = int(now_ns)
        elapsed_s = min(max((now_ns - self._last_ns) / 1e9, 0.0), 0.05)
        if elapsed_s == 0:
            return self._command.copy()
        self._last_ns = now_ns
        blend = -np.expm1(-elapsed_s / self.time_constant_s)
        self._input += blend * (target - self._input)
        desired = self._command + blend * (self._input - self._command)
        maximum_step = self.max_speed_rad_s * elapsed_s
        self._command += np.clip(desired - self._command,
                                 -maximum_step, maximum_step)
        return self._command.copy()


class RobotWriter:
    """Append complete control cycles on a writer thread without filling missing data."""

    def __init__(self, path, kinematics, capacity=2048):
        self.path = Path(path)
        self.kinematics = kinematics
        self.queue = queue.Queue(maxsize=capacity)
        self.error = None
        self.count = 0
        self.thread = None

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._write, name="robot_hdf5_writer", daemon=True)
        self.thread.start()

    def _write(self):
        import h5py

        try:
            with h5py.File(self.path, "x") as file:
                file.attrs["schema_version"] = "fr3_raw_v1"
                file.attrs["time_clock"] = "host_monotonic_ns"
                file.attrs["eef_frame"] = "fr3_link8"
                file.attrs["quaternion_order"] = "xyzw"
                file.attrs["rot6d_order"] = "first_two_rotation_columns"
                file.attrs["gripper_unit"] = "open_fraction_0_closed_1_open"
                file.attrs["command_status"] = "0 unsent; 1 interface accepted; 2 interface error; 3 unknown after call error"
                datasets = {}
                for name, (shape, dtype, _) in FIELDS.items():
                    datasets[name] = file.create_dataset(
                        name, shape=(0, *shape), maxshape=(None, *shape),
                        chunks=(256, *shape), dtype=dtype, compression="gzip")
                while True:
                    row = self.queue.get()
                    if row is None:
                        self.queue.task_done()
                        break
                    if row.get("diagnostics/observation_valid"):
                        pose, pose_6d = self.kinematics.pose(row["observations/joint_positions"])
                        row["observations/eef_pose"] = pose
                        row["observations/eef_pose_6d"] = pose_6d
                    if row.get("diagnostics/command_valid"):
                        pose, pose_6d = self.kinematics.pose(row["actions/target_joint_positions"])
                        row["actions/target_eef_pose"] = pose
                        row["actions/target_eef_pose_6d"] = pose_6d
                    index = self.count
                    for name, (shape, dtype, missing) in FIELDS.items():
                        dataset = datasets[name]
                        dataset.resize(index + 1, axis=0)
                        value = row[name] if name in row else np.full(shape, missing, dtype=dtype)
                        dataset[index] = np.asarray(value, dtype=dtype).reshape(shape)
                    self.count += 1
                    self.queue.task_done()
                file.flush()
        except Exception as exc:
            self.error = exc

    def append(self, row):
        self.check()
        try:
            self.queue.put_nowait(row)
        except queue.Full as exc:
            raise RuntimeError("Robot writer queue is full") from exc

    def check(self):
        if self.error is not None:
            raise RuntimeError("Robot writer failed") from self.error

    def close(self):
        if self.thread is None:
            return
        while self.thread.is_alive():
            try:
                self.queue.put(None, timeout=0.2)
                break
            except queue.Full:
                self.check()
        self.thread.join(timeout=60)
        if self.thread.is_alive():
            raise RuntimeError("Robot writer did not stop")
        self.check()


def record_cycle(robot, arm_stub, grip_stub, agent, kinematics, cycle_id,
                 startup_max_delta_rad=None, startup_max_error_rad=None,
                 joint_smoother=None):
    """Acquire feedback, then submit and record the exact target acknowledged by the node."""
    import polymetis_pb2

    row = {"timestamps/control_cycle_id": cycle_id}
    empty = polymetis_pb2.Empty()
    command_attempted = False
    try:
        arm = arm_stub.GetRobotState(empty, timeout=0.05)
        observation_ns = time.monotonic_ns()
        joints = np.asarray(arm.joint_positions, dtype=np.float64)
        velocities = np.asarray(arm.joint_velocities, dtype=np.float64)
        if joints.shape != (7,) or velocities.shape != (7,) or not (
                np.isfinite(joints).all() and np.isfinite(velocities).all()):
            raise ValueError("Invalid measured arm state")
        grip = grip_stub.GetState(empty, timeout=0.05)
        grip_ns = time.monotonic_ns()
        width = float(grip.width)
        if not np.isfinite(width) or width < 0:
            raise ValueError("Invalid measured gripper width")
        row.update({"timestamps/observation_ns": observation_ns,
                    "timestamps/gripper_observation_ns": grip_ns,
                    "timestamps/arm_device_sample_time_ms": int(getattr(arm, "device_sample_time_ms", -1)),
                    "timestamps/gripper_device_sample_time_ms": int(getattr(grip, "device_sample_time_ms", -1)),
                    "observations/joint_positions": joints,
                    "observations/joint_velocities": velocities,
                    "observations/gripper_state": [width],
                    "diagnostics/observation_valid": True})
    except Exception as exc:
        row["diagnostics/observation_valid"] = False
        row["diagnostics/observation_error"] = str(exc).encode("utf-8")[:256]
    try:
        mapped = np.asarray(agent.act({}), dtype=np.float64)
        if mapped.shape != (8,) or not np.isfinite(mapped).all():
            raise ValueError("Invalid mapped GELLO target")
        # The mapped and calibrated GELLO values are clipped before submission.
        target = np.r_[np.clip(mapped[:7], kinematics.lower, kinematics.upper),
                       np.clip(mapped[7], 0.0, 1.0)]
        if startup_max_delta_rad is not None:
            if not np.isfinite(startup_max_delta_rad) or startup_max_delta_rad <= 0:
                raise ValueError("Startup joint step must be positive and finite")
            if not row.get("diagnostics/observation_valid"):
                raise RuntimeError("FR3 feedback unavailable during startup transition")
            if not kinematics.within_limits(joints):
                raise RuntimeError("FR3 feedback outside joint limits during startup transition")
            if not kinematics.within_limits(mapped[:7]):
                raise RuntimeError("GELLO target outside joint limits during startup transition")
            delta = target[:7] - joints
            maximum = float(np.max(np.abs(delta)))
            if np.any(np.abs(delta) > np.broadcast_to(startup_max_error_rad, (7,))):
                raise RuntimeError("GELLO and FR3 are no longer aligned for startup")
            row["_startup_transition_complete"] = maximum <= startup_max_delta_rad
            if maximum > startup_max_delta_rad:
                target[:7] = joints + delta * (startup_max_delta_rad / maximum)
        elif joint_smoother is not None:
            target[:7] = joint_smoother.step(target[:7], time.monotonic_ns())
        before_send = time.monotonic_ns()
        command_attempted = True
        receipt = robot.command_joint_state(target)
        after_send = time.monotonic_ns()
        if not isinstance(receipt, dict):
            raise RuntimeError("Robot node did not return a command receipt")
        sent = np.asarray(receipt.get("target_joint_positions"), dtype=np.float64)
        grip_target = float(receipt.get("target_gripper", np.nan))
        grip_status = receipt.get("gripper_command_status")
        grip_command_ns = receipt.get("gripper_command_ns")
        command_ns = int(receipt["command_ns"])
        update_index = int(receipt["controller_update_idx"])
        if (sent.shape != (7,) or not np.isfinite(sent).all() or
                not np.allclose(sent, target[:7], rtol=0, atol=1e-6) or
                not np.isfinite(grip_target) or not 0 <= grip_target <= 1 or
                not kinematics.within_limits(sent) or
                not before_send <= command_ns <= after_send or update_index < 0 or
                grip_status not in ("queued", "held") or
                (grip_status == "queued") != (grip_command_ns is not None) or
                (grip_command_ns is not None and
                 not before_send <= int(grip_command_ns) <= after_send)):
            raise RuntimeError("Robot node returned a mismatched target")
        row.update({"timestamps/command_ns": command_ns,
                    "timestamps/gripper_command_ns": int(grip_command_ns or -1),
                    "actions/target_joint_positions": sent,
                    "actions/target_gripper": [grip_target],
                    "diagnostics/command_valid": True,
                    "diagnostics/command_status": 1,
                    "diagnostics/gripper_command_status": 1 if grip_status == "queued" else 0,
                    "diagnostics/controller_update_idx": update_index})
    except Exception as exc:
        row["diagnostics/command_valid"] = False
        row["diagnostics/command_status"] = 3 if command_attempted else 0
        row["diagnostics/command_error"] = str(exc).encode("utf-8")[:256]
    return row
