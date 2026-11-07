import time
from typing import Any, Dict, Optional

import numpy as np

from teleop.cameras.camera import CameraDriver
from teleop.robots.robot import Robot


class Rate:
    def __init__(self, rate: float):
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("Control rate must be positive and finite")
        self.next_tick = time.monotonic()
        self.period = 1.0 / rate

    def sleep(self) -> None:
        self.next_tick += self.period
        remaining = self.next_tick - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        else:
            self.next_tick = time.monotonic()


class RobotEnv:
    def __init__(
        self,
        robot: Robot,
        control_rate_hz: float = 100.0,
        camera_dict: Optional[Dict[str, CameraDriver]] = None,
    ) -> None:
        self._robot = robot
        self._rate = Rate(control_rate_hz)
        self._num_dofs = None
        self._camera_dict = {} if camera_dict is None else camera_dict

    def robot(self) -> Robot:
        """Get the robot object.

        Returns:
            robot: the robot object.
        """
        return self._robot

    def __len__(self):
        return 0

    def command(self, joints: np.ndarray) -> None:
        """Submit a command at the configured rate without reading sensors."""
        if self._num_dofs is None:
            self._num_dofs = self._robot.num_dofs()
        if len(joints) != self._num_dofs:
            raise ValueError(f"input:{len(joints)}, robot:{self._num_dofs}")
        self._robot.command_joint_state(joints)
        self._rate.sleep()

    def step(self, joints: np.ndarray) -> Dict[str, Any]:
        """Submit a command and return the current observation."""
        self.command(joints)
        return self.get_obs()

    def get_obs(self) -> Dict[str, Any]:
        """Get observation from the environment.

        Returns:
            obs: observation from the environment.
        """
        observations = {}
        for name, camera in self._camera_dict.items():
            image, depth = camera.read()
            observations[f"{name}_rgb"] = image
            if isinstance(depth, dict):
                observations[f"{name}_frame_metadata"] = depth
            else:
                observations[f"{name}_depth"] = depth

        robot_obs = self._robot.get_observations()
        assert "joint_positions" in robot_obs
        assert "joint_velocities" in robot_obs
        assert "ee_pos_quat" in robot_obs
        observations.update(robot_obs)
        observations["observation_host_monotonic_ns"] = time.monotonic_ns()
        return observations


def main() -> None:
    pass


if __name__ == "__main__":
    main()
