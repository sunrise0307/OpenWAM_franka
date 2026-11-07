"""Display live GELLO/FR3 joint differences before recording an episode."""

import argparse
import sys
import time

import numpy as np


MAX_JOINT_ERRORS_RAD = (0.30,) * 7
POLL_INTERVAL_S = 0.25
STABLE_SAMPLES = 5


def alignment_status(gello, fr3, lower, upper, max_error=MAX_JOINT_ERRORS_RAD):
    values = [np.asarray(value, dtype=float) for value in (gello, fr3, lower, upper)]
    if any(value.shape != (7,) or not np.isfinite(value).all() for value in values):
        raise ValueError("Expected seven finite GELLO, FR3 and joint-limit values")
    gello, fr3, lower, upper = values
    within_limits = (gello >= lower) & (gello <= upper)
    aligned = np.abs(gello - fr3) <= np.broadcast_to(max_error, (7,))
    return within_limits & aligned, within_limits


def main():
    parser = argparse.ArgumentParser(description="采集前手动对齐 GELLO 和 FR3")
    parser.add_argument("--port", required=True, help="GELLO 的 /dev/serial/by-id 路径")
    args = parser.parse_args()

    from polymetis import RobotInterface
    from teleop.agents.teleop_agent import TeleopAgent

    robot = RobotInterface(ip_address="127.0.0.1", port=50051)
    agent = None
    try:
        initial_fr3 = np.asarray(robot.get_robot_state().joint_positions, dtype=float)
        if initial_fr3.shape != (7,) or not np.isfinite(initial_fr3).all():
            raise ValueError("Invalid initial FR3 joint feedback")
        limits = robot.robot_model.get_joint_angle_limits()
        lower = limits[0].detach().cpu().numpy()
        upper = limits[1].detach().cpu().numpy()
        # Match the collector's 2*pi branch choice for multi-turn GELLO encoders.
        agent = TeleopAgent(port=args.port, start_joints=np.r_[initial_fr3, 0.0])
        consecutive = 0
        while consecutive < STABLE_SAMPLES:
            fr3 = np.asarray(robot.get_robot_state().joint_positions, dtype=float)
            gello = np.asarray(agent.act({}), dtype=float)
            if gello.shape != (8,) or not np.isfinite(gello).all():
                raise ValueError("Invalid GELLO joint feedback")
            ready, within_limits = alignment_status(gello[:7], fr3, lower, upper)
            consecutive = consecutive + 1 if ready.all() else 0
            if sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print("手动调整 GELLO；采集器不会在此阶段移动 FR3。")
            print("需在 FR3 限位内；每个关节差值 <= 0.30 rad；"
                  f"连续 {STABLE_SAMPLES} 次合格后开始采集。Ctrl+C 退出。")
            print("关节     FR3(rad)   GELLO(rad)   差值(rad)   阈值(rad)    限位(rad)        状态")
            for index in range(7):
                state = "就绪" if ready[index] else ("超限" if not within_limits[index] else "未对齐")
                print(f"J{index + 1}       {fr3[index]:+8.3f}    {gello[index]:+8.3f}"
                      f"    {gello[index] - fr3[index]:+8.3f}     "
                      f"{MAX_JOINT_ERRORS_RAD[index]:.2f}    "
                      f"[{lower[index]:+.2f}, {upper[index]:+.2f}]   {state}")
            print(f"稳定计数: {consecutive}/{STABLE_SAMPLES}", flush=True)
            if consecutive < STABLE_SAMPLES:
                time.sleep(POLL_INTERVAL_S)
        print("GELLO 已对齐，开始当前 episode。")
    finally:
        if agent is not None:
            agent._robot._driver.close()
        robot.channel.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已取消 GELLO 对齐。", file=sys.stderr)
        raise SystemExit(130)
