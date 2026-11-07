import datetime
import glob
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tyro

from teleop.agents.agent import BimanualAgent, DummyAgent
from teleop.agents.teleop_agent import TeleopAgent
from teleop.data_utils.format_obs import save_frame
from teleop.env import RobotEnv
from teleop.robots.robot import PrintRobot
from teleop.zmq_core.robot_node import ZMQClientRobot
from teleop.zmq_core.camera_node import ZMQClientCamera


def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


@dataclass
class Args:
    agent: str = "none"
    tele_port: int = 6001
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hostname: str = "127.0.0.1"
    robot_type: str = None  # only needed for quest agent or spacemouse agent
    hz: int = 100
    start_joints: Optional[Tuple[float, ...]] = None

    teleop_port: Optional[str] = None
    mock: bool = False
    use_save_interface: bool = False
    data_dir: str = "~/bc_data"
    bimanual: bool = False
    verbose: bool = False
    capture_rgb: bool = False


class TeleopCommandSmoother:
    """Turn sparse master-arm readings into bounded joint commands."""

    def __init__(self, time_constant_s=0.03, max_speed_rad_s=1.5):
        if (not np.isfinite(time_constant_s) or time_constant_s <= 0 or
                not np.isfinite(max_speed_rad_s) or max_speed_rad_s <= 0):
            raise ValueError("Smoothing settings must be positive and finite")
        self.time_constant_s = float(time_constant_s)
        self.max_speed_rad_s = float(max_speed_rad_s)
        self.input = None
        self.command = None
        self.last_ns = None

    def reset(self, joints, now_ns):
        joints = np.asarray(joints, dtype=np.float64)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError("Expected seven finite starting joints")
        self.input = joints.copy()
        self.command = joints.copy()
        self.last_ns = int(now_ns)

    def step(self, action, now_ns):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError("Expected seven joints and one gripper target")
        if self.last_ns is None:
            raise RuntimeError("Smoother has no starting command")
        now_ns = int(now_ns)
        elapsed_s = min(max((now_ns - self.last_ns) / 1e9, 0.0), 0.05)
        if elapsed_s:
            self.last_ns = now_ns
            blend = -np.expm1(-elapsed_s / self.time_constant_s)
            self.input += blend * (action[:7] - self.input)
            desired = self.command + blend * (self.input - self.command)
            maximum_step = self.max_speed_rad_s * elapsed_s
            self.command += np.clip(desired - self.command,
                                    -maximum_step, maximum_step)
        return np.r_[self.command, action[7]]


def record_teleop_command(env, action, save_path, timestamp):
    """Pair the current observation with the command sent after it."""
    observation = env.get_obs()
    env.command(action)
    save_frame(save_path, timestamp, observation, action)
    return observation


def check_startup_alignment(action, joints, maximum_error=0.5):
    difference = np.asarray(action[:7]) - np.asarray(joints[:7])
    if difference.shape != (7,) or not np.isfinite(difference).all():
        raise ValueError("Invalid startup joint positions")
    outside = np.flatnonzero(np.abs(difference) > maximum_error)
    if outside.size:
        details = ", ".join(f"joint {index + 1}: {difference[index]:+.3f}"
                            for index in outside)
        raise ValueError("GELLO and FR3 differ by over 0.5 radians: " + details)


def main(args):
    if args.mock:
        robot_client = PrintRobot(8, dont_print=True)
        camera_clients = {}
    else:
        camera_clients = (
            {
                "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
                "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
            } if args.capture_rgb else {}
        )
        robot_client = ZMQClientRobot(port=args.tele_port, host=args.hostname)
    env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients)

    if args.bimanual:
        if args.agent == "teleop":
            # dynamixel control box port map (to distinguish left and right teleop)
            right = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBG6A-if00-port0"
            left = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBEIA-if00-port0"
            left_agent = TeleopAgent(port=left)
            right_agent = TeleopAgent(port=right)
            agent = BimanualAgent(left_agent, right_agent)
        elif args.agent == "quest":
            from teleop.agents.quest_agent import SingleArmQuestAgent

            left_agent = SingleArmQuestAgent(robot_type=args.robot_type, which_hand="l")
            right_agent = SingleArmQuestAgent(
                robot_type=args.robot_type, which_hand="r"
            )
            agent = BimanualAgent(left_agent, right_agent)
            # raise NotImplementedError
        elif args.agent == "spacemouse":
            from teleop.agents.spacemouse_agent import SpacemouseAgent

            left_path = "/dev/hidraw0"
            right_path = "/dev/hidraw1"
            left_agent = SpacemouseAgent(
                robot_type=args.robot_type, device_path=left_path, verbose=args.verbose
            )
            right_agent = SpacemouseAgent(
                robot_type=args.robot_type,
                device_path=right_path,
                verbose=args.verbose,
                invert_button=True,
            )
            agent = BimanualAgent(left_agent, right_agent)
        else:
            raise ValueError(f"Invalid agent name for bimanual: {args.agent}")

        # System setup specific. This reset configuration works well on our setup. If you are mounting the robot
        # differently, you need a separate reset joint configuration.
        reset_joints_left = np.deg2rad([0, -90, -90, -90, 90, 0, 0])
        reset_joints_right = np.deg2rad([0, -90, 90, -90, -90, 0, 0])
        reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
        curr_joints = env.get_obs()["joint_positions"]
        max_delta = (np.abs(curr_joints - reset_joints)).max()
        steps = min(int(max_delta / 0.01), 100)

        for jnt in np.linspace(curr_joints, reset_joints, steps):
            env.step(jnt)
    else:
        if args.agent == "teleop":
            teleop_port = args.teleop_port
            if teleop_port is None:
                usb_ports = glob.glob("/dev/serial/by-id/*")
                print(f"Found {len(usb_ports)} ports")
                if len(usb_ports) > 0:
                    teleop_port = usb_ports[0]
                    print(f"using port {teleop_port}")
                else:
                    raise ValueError(
                        "No teleop port found, please specify one or plug in teleop"
                    )
            if args.start_joints is None:
                reset_joints = np.deg2rad(
                    [0, -90, 90, -90, -90, 0, 0]
                )  # Change this to your own reset joints
            else:
                reset_joints = args.start_joints
                reset_joints = np.array(reset_joints)
            agent = TeleopAgent(port=teleop_port, start_joints=args.start_joints)
            curr_joints = env.get_obs()["joint_positions"]
            curr_joints = np.array(curr_joints)
            if reset_joints.shape == curr_joints.shape:
                max_delta = (np.abs(curr_joints - reset_joints)).max()
                steps = min(int(max_delta / 0.01), 100)

                for jnt in np.linspace(curr_joints, reset_joints, steps):
                    env.step(jnt)
                    time.sleep(0.001)
        elif args.agent == "quest":
            from teleop.agents.quest_agent import SingleArmQuestAgent

            agent = SingleArmQuestAgent(robot_type=args.robot_type, which_hand="l")
        elif args.agent == "spacemouse":
            from teleop.agents.spacemouse_agent import SpacemouseAgent

            agent = SpacemouseAgent(robot_type=args.robot_type, verbose=args.verbose)
        elif args.agent == "dummy" or args.agent == "none":
            agent = DummyAgent(num_dofs=robot_client.num_dofs())
        elif args.agent == "policy":
            raise NotImplementedError("add your imitation policy here if there is one")
        else:
            raise ValueError("Invalid agent name")

    # going to start position
    print("Going to start position")
    start_pos = agent.act(env.get_obs())
    obs = env.get_obs()
    joints = obs["joint_positions"]
    joints = np.array(joints)

    abs_deltas = np.abs(start_pos - joints)
    id_max_joint_delta = np.argmax(abs_deltas)

    max_joint_delta = 0.8
    if abs_deltas[id_max_joint_delta] > max_joint_delta:
        id_mask = abs_deltas > max_joint_delta
        print()

        ids = np.arange(len(id_mask))[id_mask]
        for i, delta, joint, current_j in zip(
            ids,
            abs_deltas[id_mask],
            start_pos[id_mask],
            joints[id_mask],
        ):
            print(
                f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
            )
        return

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(
        joints
    ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    max_delta = 0.05
    last_startup_command = None
    for _ in range(25):
        startup_tick = time.monotonic()
        obs = env.get_obs()
        command_joints = agent.act(obs)
        current_joints = obs["joint_positions"]
        delta = command_joints - current_joints
        max_joint_delta = np.abs(delta).max()
        if max_joint_delta > max_delta:
            delta = delta / max_joint_delta * max_delta
        last_startup_command = current_joints + delta
        env.step(last_startup_command)
        if args.agent == "teleop":
            time.sleep(max(0.0, startup_tick + 0.05 - time.monotonic()))

    obs = env.get_obs()
    joints = obs["joint_positions"]
    action = agent.act(obs)
    check_startup_alignment(action, joints)

    if args.use_save_interface:
        from teleop.data_utils.keyboard_interface import KBReset

        kb_interface = KBReset()

    print_color("\nStart 🚀🚀🚀", color="green", attrs=("bold",))

    smoother = None
    if args.agent == "teleop" and not args.bimanual:
        smoother = TeleopCommandSmoother()
        smoother.reset(last_startup_command[:7], time.monotonic_ns())
    save_path = None
    last_saved_ns = None
    start_time = time.time()
    while True:
        num = time.time() - start_time
        message = f"\rTime passed: {round(num, 2)}          "
        print_color(
            message,
            color="white",
            attrs=("bold",),
            end="",
            flush=True,
        )
        action = agent.act(obs)
        if smoother is not None:
            action = smoother.step(action, time.monotonic_ns())
        dt = datetime.datetime.now()
        state = "normal"
        if args.use_save_interface:
            state = kb_interface.update()
            if state == "start":
                dt_time = datetime.datetime.now()
                save_path = (
                    Path(args.data_dir).expanduser()
                    / args.agent
                    / dt_time.strftime("%m%d_%H%M%S")
                )
                save_path.mkdir(parents=True, exist_ok=True)
                print(f"Saving to {save_path}")
                last_saved_ns = None
            elif state == "save":
                assert save_path is not None, "something went wrong"
            elif state == "normal":
                save_path = None
            else:
                raise ValueError(f"Invalid state {state}")
        if smoother is None:
            if args.use_save_interface and state == "save":
                save_frame(save_path, dt, obs, action)
            obs = env.step(action)
        elif state == "save" and (
                last_saved_ns is None or time.monotonic_ns() - last_saved_ns >= 50_000_000):
            obs = record_teleop_command(env, action, save_path, dt)
            last_saved_ns = time.monotonic_ns()
        else:
            env.command(action)


if __name__ == "__main__":
    main(tyro.cli(Args))
