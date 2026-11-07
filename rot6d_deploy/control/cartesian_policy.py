"""A Cartesian tracker with cycle-level speed limits and a target watchdog."""

from typing import Dict

import torch
import torchcontrol as toco


class WatchdogCartesianImpedanceControl(toco.policies.CartesianImpedanceControl):
    """Latch the measured pose if target updates stop, even if Python exits."""

    def __init__(self, joint_pos_current, Kp, Kd, robot_model,
                 watchdog_ticks, control_hz=1000,
                 max_linear_velocity_m_s=0.8, max_angular_velocity_rad_s=2.5,
                 ignore_gravity=True):
        super().__init__(joint_pos_current, Kp, Kd, robot_model,
                         ignore_gravity=ignore_gravity)
        if watchdog_ticks < 1:
            raise ValueError("watchdog_ticks must be positive")
        if control_hz <= 0 or max_linear_velocity_m_s <= 0 or max_angular_velocity_rad_s <= 0:
            raise ValueError("control frequency and reference speeds must be positive")
        self.watchdog_ticks = int(watchdog_ticks)
        self.max_position_step = float(max_linear_velocity_m_s / control_hz)
        self.max_rotation_step = float(max_angular_velocity_rad_s / control_hz)
        self.heartbeat = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer("last_heartbeat", torch.full((1,), -1.0))
        self.register_buffer("stale_ticks", torch.zeros(1, dtype=torch.int64))
        self.register_buffer("hold_latched", torch.zeros(1, dtype=torch.bool))
        self.register_buffer("held_position", torch.zeros(3))
        self.register_buffer("held_quaternion", torch.tensor([0.0, 0.0, 0.0, 1.0]))
        self.register_buffer("reference_position", self.ee_pos_desired.detach().clone())
        self.register_buffer("reference_quaternion", self.ee_quat_desired.detach().clone())

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        joint_position = state_dict["joint_positions"]
        joint_velocity = state_dict["joint_velocities"]
        measured_position, measured_quaternion = self.robot_model.forward_kinematics(joint_position)

        if bool((self.heartbeat != self.last_heartbeat).any()):
            self.last_heartbeat.copy_(self.heartbeat)
            if not bool(self.hold_latched[0]):
                self.stale_ticks.zero_()
        else:
            self.stale_ticks.add_(1)

        if not bool(self.hold_latched[0]) and int(self.stale_ticks[0]) >= self.watchdog_ticks:
            self.held_position.copy_(measured_position)
            self.held_quaternion.copy_(measured_quaternion)
            self.hold_latched.fill_(1)

        if bool(self.hold_latched[0]):
            desired_position = self.held_position
            desired_quaternion = self.held_quaternion
        else:
            displacement = self.ee_pos_desired - self.reference_position
            distance = torch.linalg.vector_norm(displacement)
            if bool(distance > self.max_position_step):
                self.reference_position.add_(displacement * (self.max_position_step / distance))
            else:
                self.reference_position.copy_(self.ee_pos_desired)

            target_quaternion = self.ee_quat_desired / torch.linalg.vector_norm(self.ee_quat_desired)
            dot = torch.dot(self.reference_quaternion, target_quaternion)
            if bool(dot < 0):
                target_quaternion = -target_quaternion
                dot = -dot
            half_angle = torch.acos(torch.clamp(dot, 0.0, 1.0))
            if bool(2.0 * half_angle > self.max_rotation_step):
                fraction = self.max_rotation_step / (2.0 * half_angle)
                sine = torch.sin(half_angle)
                interpolated = (
                    torch.sin((1.0 - fraction) * half_angle) / sine * self.reference_quaternion
                    + torch.sin(fraction * half_angle) / sine * target_quaternion
                )
                self.reference_quaternion.copy_(interpolated / torch.linalg.vector_norm(interpolated))
            else:
                self.reference_quaternion.copy_(target_quaternion)
            desired_position = self.reference_position
            desired_quaternion = self.reference_quaternion

        jacobian = self.robot_model.compute_jacobian(joint_position)
        measured_twist = jacobian @ joint_velocity
        feedback_wrench = self.pose_pd(
            measured_position,
            measured_quaternion,
            measured_twist,
            desired_position,
            desired_quaternion,
            torch.cat([self.ee_vel_desired, self.ee_rvel_desired]),
        )
        feedback_torque = jacobian.T @ feedback_wrench
        feedforward_torque = self.invdyn(
            joint_position, joint_velocity, torch.zeros_like(joint_position)
        )
        return {"joint_torques": feedback_torque + feedforward_torque}
