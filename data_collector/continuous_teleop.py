"""Keep joint teleoperation running while independent raw episodes are recorded."""

import shutil
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from align_gello import MAX_JOINT_ERRORS_RAD
from camera_capture import ColorRecorder, SessionCamera
from collector import FR3_URDF_PATH, atomic_json, connect_grpc, sha256_file
from fr3_schema import Fr3Kinematics, JointTargetSmoother, RobotWriter, record_cycle


STARTUP_MAX_DELTA_RAD = 0.05
STARTUP_TIMEOUT_S = 10.0
JOINT_SMOOTHING_TIME_S = 0.03
MAX_JOINT_SPEED_RAD_S = 1.5


class ContinuousTeleop:
    def __init__(self, args, manifest):
        self.args = args
        self.manifest = Path(manifest)
        self.stop = threading.Event()
        self.control_ready = threading.Event()
        self.lock = threading.Lock()
        self.active = None
        self.thread = None
        self.error = None
        self.channels = []
        self.camera_sources = {}
        self.cycle_id = 0

    def start(self):
        import zmq
        from teleop.agents.teleop_agent import TeleopAgent
        from teleop.zmq_core.robot_node import ZMQClientRobot

        args = self.args
        if getattr(args, "host", "127.0.0.1") not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("Robot node and collector must use the same host monotonic clock")
        self.kinematics = Fr3Kinematics(getattr(args, "urdf", FR3_URDF_PATH),
                                        getattr(args, "ee_link", "fr3_link8"))
        arm_channel, grip_channel, self.arm_stub, self.grip_stub, self.arm_meta, self.grip_meta = (
            connect_grpc(getattr(args, "host", "127.0.0.1"),
                         getattr(args, "arm_port", 50051),
                         getattr(args, "gripper_port", 50052),
                         getattr(args, "urdf", FR3_URDF_PATH),
                         getattr(args, "ee_link", "fr3_link8")))
        self.channels = [arm_channel, grip_channel]
        self.robot = ZMQClientRobot(port=getattr(args, "teleop_port", 6001),
                                    host=getattr(args, "host", "127.0.0.1"))
        self.robot._socket.setsockopt(zmq.RCVTIMEO, 1000)
        self.robot._socket.setsockopt(zmq.SNDTIMEO, 1000)
        feedback = self.robot.get_observations()
        joints = np.asarray(feedback["joint_positions"][:7], dtype=np.float64)
        if self.robot.num_dofs() != 8 or joints.shape != (7,) or not np.isfinite(joints).all():
            raise RuntimeError("The FR3 joint-control node did not return valid feedback")
        self.agent = TeleopAgent(port=args.gello_port, start_joints=np.r_[joints, 0.0])
        preview_dir = (Path(args.output_dir).expanduser().resolve() / "camera_previews" /
                       (time.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]))
        preview_dir.mkdir(parents=True, exist_ok=False)
        for slot, serial in (("head_camera", args.head_serial),
                             ("left_wrist_camera", args.wrist_serial)):
            source = SessionCamera(slot, serial, preview_dir / (slot + ".jpg"))
            self.camera_sources[slot] = source
            source.start()
        for source in self.camera_sources.values():
            if not source.ready.wait(timeout=source.WARMUP_SECONDS + 5):
                source.check()
                raise RuntimeError("No color frames received from " + source.slot)
        self.thread = threading.Thread(target=self._control, name="fr3_control", daemon=True)
        self.thread.start()
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while not self.control_ready.wait(timeout=0.1):
            self.check()
            if time.monotonic() >= deadline:
                self.stop.set()
                raise RuntimeError("FR3 startup transition did not complete in time")
        self.check()
        for source in self.camera_sources.values():
            print(f"{source.slot}: {source.preview_path}", flush=True)

    def check(self):
        if self.error is not None:
            raise RuntimeError("Teleoperation stopped: " + str(self.error)) from self.error
        for source in self.camera_sources.values():
            source.check()
        with self.lock:
            active = self.active
        if active:
            active["writer"].check()
            for camera in active["cameras"]:
                camera.check()

    def _control(self):
        period_ns = round(1_000_000_000 / self.args.teleop_hz)
        tick = time.monotonic_ns()
        starting = True
        smoother = JointTargetSmoother(JOINT_SMOOTHING_TIME_S, MAX_JOINT_SPEED_RAD_S)
        try:
            while not self.stop.is_set():
                startup_options = ({"startup_max_delta_rad": STARTUP_MAX_DELTA_RAD,
                                    "startup_max_error_rad": MAX_JOINT_ERRORS_RAD}
                                   if starting else {"joint_smoother": smoother})
                row = record_cycle(self.robot, self.arm_stub, self.grip_stub,
                                   self.agent, self.kinematics, self.cycle_id,
                                   **startup_options)
                with self.lock:
                    if self.active is not None:
                        self.active["writer"].append(row)
                        self.active["count"] += 1
                        self.active["invalid_observations"] += int(
                            not row["diagnostics/observation_valid"])
                        self.active["invalid_commands"] += int(
                            not row["diagnostics/command_valid"])
                self.cycle_id += 1
                if not row["diagnostics/command_valid"]:
                    if starting:
                        detail = row.get("diagnostics/command_error", b"")
                        if isinstance(detail, bytes):
                            detail = detail.decode("utf-8", errors="replace")
                        raise RuntimeError("FR3 startup transition failed: " + str(detail))
                    raise RuntimeError("FR3 command failed or lacked a verifiable receipt")
                if starting and row.get("_startup_transition_complete"):
                    smoother.reset(row["actions/target_joint_positions"],
                                   time.monotonic_ns())
                    starting = False
                    self.control_ready.set()
                tick += period_ns
                now = time.monotonic_ns()
                if now > tick:
                    missed = (now - tick) // period_ns + 1
                    with self.lock:
                        if self.active is not None:
                            for _ in range(missed):
                                self.active["writer"].append({
                                    "timestamps/control_cycle_id": self.cycle_id,
                                    "diagnostics/observation_error": b"control deadline missed",
                                    "diagnostics/command_error": b"control deadline missed"})
                                self.active["count"] += 1
                                self.active["missed_cycles"] += 1
                                self.cycle_id += 1
                    tick += missed * period_ns
                self.stop.wait(max(0, (tick - time.monotonic_ns()) / 1e9))
        except Exception as exc:
            self.error = exc
            self.stop.set()

    def begin(self, row, episode_id):
        self.check()
        args = self.args
        episode = Path(args.output_dir).expanduser().resolve() / "raw" / args.task_id / episode_id
        episode.mkdir(parents=True, exist_ok=False)
        dataset_metadata = episode.parent / "dataset_metadata.json"
        if not dataset_metadata.exists():
            atomic_json(dataset_metadata, {
                "schema_version": "fr3_raw_v1", "task_id": args.task_id,
                "prompt": args.instruction, "control_hz_target": args.teleop_hz,
                "camera_fps_target": 30, "capture_module": "continuous_teleop"})
        active = {"episode": episode, "episode_id": episode_id,
                  "stop": threading.Event(), "count": 0, "missed_cycles": 0,
                  "invalid_observations": 0, "invalid_commands": 0, "cameras": []}
        try:
            if args.calibration_dir:
                shutil.copytree(args.calibration_dir, episode / "calibration")
            active["writer"] = RobotWriter(episode / "robot.hdf5", self.kinematics)
            active["writer"].start()
            for slot, serial in (("head_camera", args.head_serial),
                                 ("left_wrist_camera", args.wrist_serial)):
                camera = ColorRecorder(slot, serial, episode, active["stop"],
                                       source=self.camera_sources[slot])
                active["cameras"].append(camera)
                camera.start()
            for camera in active["cameras"]:
                if not camera.first.wait(timeout=3):
                    camera.check()
                    raise RuntimeError("No recorded color frame from " + camera.name)
            project = Path(__file__).resolve().parents[1]
            calibration_hashes = {
                str(path.relative_to(episode / "calibration")): sha256_file(path)
                for path in (episode / "calibration").rglob("*") if path.is_file()
            } if args.calibration_dir else {}
            metadata = {
                "schema_version": "fr3_raw_v1", "episode_id": episode_id,
                "task_id": args.task_id, "planned_id": row["planned_episode_id"],
                "planned_order": row["planned_order"], "planned_split": row["split"],
                "prompt": args.instruction, "instruction": args.instruction,
                "success_criterion": args.success_criterion, "session_id": args.session_id,
                "operator": args.operator, "operator_success": None,
                "acceptance_status": "pending",
                "start_host_monotonic_ns": time.monotonic_ns(),
                "robot_model": "FR3", "controller_type": "joint_impedance",
                "joint_order": [f"fr3_joint{i}" for i in range(1, 8)],
                "eef_frame": getattr(args, "ee_link", "fr3_link8"),
                "quaternion_order": "xyzw", "rot6d_order": "first_two_rotation_columns",
                "fk_model_sha256": sha256_file(getattr(args, "urdf", FR3_URDF_PATH)),
                "joint_lower_rad": self.kinematics.lower.tolist(),
                "joint_upper_rad": self.kinematics.upper.tolist(),
                "gripper_state_unit": "width_m",
                "gripper_command_unit": "open_fraction_0_closed_1_open",
                "gripper_command_semantics": "changed targets are queued at most 20 times per second; a held row repeats the latest queued target; completion is unverified",
                "gripper_max_width_m": self.grip_meta.max_width,
                "gello_mapping": "TeleopAgent DynamixelRobot signs, offsets and startup branch; two-stage command smoothing",
                "gello_mapping_sha256": sha256_file(
                    project / "frankateleop/teleop/teleop/agents/teleop_agent.py"),
                "gello_driver_sha256": sha256_file(
                    project / "frankateleop/teleop/teleop/robots/dynamixel.py"),
                "fr3_controller_sha256": sha256_file(
                    project / "frankateleop/teleop/teleop/robots/fr3.py"),
                "collector_sha256": sha256_file(Path(__file__)),
                "command_smoother_sha256": sha256_file(
                    project / "data_collector/fr3_schema.py"),
                "gello_port": args.gello_port,
                "calibration_path": "calibration" if args.calibration_dir else None,
                "calibration_checksums": calibration_hashes,
                "control_hz_target": args.teleop_hz,
                "joint_smoothing_time_s": JOINT_SMOOTHING_TIME_S,
                "max_joint_speed_rad_s": MAX_JOINT_SPEED_RAD_S,
                "state_read_semantics": "GetRobotState and GetState may return cached samples; host times mark response receipt",
                "command_time_semantics": "robot node time immediately after arm interface acknowledgement",
                "cameras": {camera.name: camera.profile_info for camera in active["cameras"]},
                "camera_extrinsics": {camera.name: None for camera in active["cameras"]},
                "camera_calibration_version": None,
            }
            active["metadata"] = metadata
            atomic_json(episode / "metadata.json", metadata)
            with self.lock:
                if self.active is not None:
                    raise RuntimeError("An episode is already recording")
                self.active = active
            return episode
        except BaseException:
            self.finish(discard=True, active=active)
            raise

    def finish(self, discard=False, active=None):
        with self.lock:
            active = active or self.active
            if self.active is active:
                self.active = None
        if active is None:
            return None
        episode = active["episode"]
        active["stop"].set()
        failures = []
        for camera in active["cameras"]:
            try:
                camera.close()
            except Exception as exc:
                failures.append(str(exc))
        if "writer" in active:
            try:
                active["writer"].close()
            except Exception as exc:
                failures.append(str(exc))
        if discard:
            shutil.rmtree(episode)
            return None
        if self.error:
            failures.append(str(self.error))
        if active["count"] == 0:
            failures.append("No robot control cycles were recorded")
        if any(camera.count == 0 for camera in active["cameras"]):
            failures.append("One or more camera streams are empty")
        metadata = active["metadata"]
        metadata.update(end_host_monotonic_ns=time.monotonic_ns(),
                        control_cycles=active["count"],
                        invalid_observations=active["invalid_observations"],
                        invalid_commands=active["invalid_commands"],
                        missed_cycles=active["missed_cycles"],
                        camera_frames={camera.name: camera.count for camera in active["cameras"]},
                        camera_drops={camera.name: camera.dropped + camera.device_missing
                                      for camera in active["cameras"]})
        atomic_json(episode / "metadata.json", metadata)
        atomic_json(episode / "summary.json", {
            "episode_id": active["episode_id"],
            "status": "failed" if failures else "raw_unverified",
            "failure": "; ".join(failures) if failures else None,
            "control_commands": active["count"],
            "invalid_observations": active["invalid_observations"],
            "invalid_commands": active["invalid_commands"],
            "missed_cycles": active["missed_cycles"],
            "camera_frames": metadata["camera_frames"]})
        checksums = {str(path.relative_to(episode)): sha256_file(path)
                     for path in episode.rglob("*") if path.is_file() and path.name != "checksums.json"}
        atomic_json(episode / "checksums.json", checksums)
        if failures:
            raise RuntimeError("Episode failed: " + "; ".join(failures))
        return episode

    def close(self):
        self.stop.set()
        try:
            if self.thread is not None:
                self.thread.join(timeout=5)
                if self.thread.is_alive():
                    raise RuntimeError("Robot control thread did not stop")
            if self.active is not None:
                self.finish(discard=True)
        finally:
            for source in self.camera_sources.values():
                source.close()
            if getattr(self, "agent", None) is not None:
                self.agent._robot._driver.close()
            if getattr(self, "robot", None) is not None:
                self.robot._socket.close()
            for channel in self.channels:
                channel.close()
