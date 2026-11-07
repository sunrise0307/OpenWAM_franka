import os
from pathlib import Path
import unittest

import numpy as np

from rot6d_deploy.control.geometry import quaternion_distance


@unittest.skipUnless(os.environ.get("POLYMETIS_BUILD_DIR"),
                     "Polymetis compiled libraries are not configured")
class CartesianPolicyTests(unittest.TestCase):
    @staticmethod
    def model():
        import torchcontrol as toco

        root = Path(__file__).resolve().parents[2]
        model_path = root / "frankateleop/polymetis/polymetis/python/polymetis/data/fr3/fr3.urdf"
        return toco.models.RobotModelPinocchio(str(model_path), "fr3_link8")

    def test_scripted_update_and_independent_timeout_hold(self):
        import torch

        from rot6d_deploy.control.cartesian_policy import WatchdogCartesianImpedanceControl

        policy = WatchdogCartesianImpedanceControl(
            joint_pos_current=torch.zeros(7),
            Kp=torch.tensor([750., 750., 750., 15., 15., 15.]),
            Kd=torch.tensor([37., 37., 37., 2., 2., 2.]),
            robot_model=self.model(),
            watchdog_ticks=3,
        )
        scripted = torch.jit.script(policy)
        scripted.update({
            "ee_pos_desired": torch.tensor([0.5, 0.0, 0.3]),
            "ee_quat_desired": torch.tensor([0.0, 0.0, 0.0, 1.0]),
            "heartbeat": torch.tensor([1.0]),
        })
        self.assertAlmostEqual(float(scripted.ee_pos_desired[0]), 0.5)
        state = {"joint_positions": torch.zeros(7), "joint_velocities": torch.zeros(7)}
        for _ in range(4):
            result = scripted(state)
        self.assertEqual(tuple(result["joint_torques"].shape), (7,))
        self.assertTrue(bool(scripted.hold_latched[0]))
        scripted.update({"heartbeat": torch.tensor([2.0])})
        scripted(state)
        self.assertTrue(bool(scripted.hold_latched[0]))

    def test_scripted_reference_reaches_full_target_at_control_rate(self):
        import math
        import torch

        from rot6d_deploy.control.cartesian_policy import WatchdogCartesianImpedanceControl

        policy = WatchdogCartesianImpedanceControl(
            joint_pos_current=torch.zeros(7),
            Kp=torch.tensor([750., 750., 750., 15., 15., 15.]),
            Kd=torch.tensor([37., 37., 37., 2., 2., 2.]),
            robot_model=self.model(), watchdog_ticks=100,
        )
        scripted = torch.jit.script(policy)
        start_position = scripted.reference_position.detach().clone()
        start_quaternion = scripted.reference_quaternion.detach().cpu().numpy()
        half_angle = 0.06 / 2
        rotation = np.array([math.sin(half_angle), 0, 0, math.cos(half_angle)])
        x, y, z, w = start_quaternion
        a, b, c, d = rotation
        target_quaternion = np.array([
            w * a + x * d + y * c - z * b,
            w * b - x * c + y * d + z * a,
            w * c + x * b - y * a + z * d,
            w * d - x * a - y * b - z * c,
        ])
        target_position = start_position + torch.tensor([0.0209, 0, 0])
        scripted.update({
            "ee_pos_desired": target_position,
            "ee_quat_desired": torch.tensor(target_quaternion, dtype=torch.float32),
            "heartbeat": torch.tensor([1.0]),
        })
        state = {"joint_positions": torch.zeros(7), "joint_velocities": torch.zeros(7)}
        scripted(state)
        self.assertLessEqual(float(torch.linalg.vector_norm(scripted.reference_position - start_position)), 0.000801)
        self.assertLessEqual(quaternion_distance(start_quaternion, scripted.reference_quaternion.detach().numpy()), 0.0026)
        for _ in range(35):
            scripted(state)
        np.testing.assert_allclose(scripted.reference_position.detach().numpy(), target_position.numpy(), atol=1e-5)
        self.assertLess(quaternion_distance(scripted.reference_quaternion.detach().numpy(), target_quaternion), 1e-3)


if __name__ == "__main__":
    unittest.main()
