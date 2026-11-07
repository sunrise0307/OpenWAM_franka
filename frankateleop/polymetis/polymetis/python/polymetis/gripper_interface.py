# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import math
import threading
import queue

import grpc

import polymetis_pb2
import polymetis_pb2_grpc


log = logging.getLogger(__name__)

EMPTY = polymetis_pb2.Empty()


class GripperInterface:
    """Gripper interface class to initialize a connection to a gRPC gripper server.

    Args:
        ip_address: IP address of the gRPC-based gripper server.
        port: Port to connect to on the IP address.
    """

    def __init__(self, ip_address: str = "localhost", port: int = 50052):
        # Connect to server
        self.channel = grpc.insecure_channel(f"{ip_address}:{port}")
        self.grpc_connection = polymetis_pb2_grpc.GripperServerStub(self.channel)

        # Get metadata
        try:
            self.metadata = self.grpc_connection.GetRobotClientMetadata(EMPTY)
        except grpc.RpcError:
            log.warning("Metadata unavailable from server.")

        # Execute commands from cache in separate thread
        self._command_thr = threading.Thread(
            target=self._command_executor,
            daemon=True,
        )

        self._command_queue = queue.Queue(maxsize=1)
        self._command_thr.start()

    def _command_executor(self):
        while True:
            command, msg, completed, result = self._command_queue.get()
            try:
                command(msg, timeout=1.0)
            except Exception as exc:
                result.append(exc)
                log.exception("Gripper command RPC failed")
            finally:
                completed.set()
                self._command_queue.task_done()

    def _send_gripper_command(self, command, msg, blocking: bool = True) -> None:
        completed = threading.Event()
        result = []
        self._command_queue.put((command, msg, completed, result), timeout=1.0)

        if blocking:
            completed.wait()
            if result:
                raise result[0]

    def get_state(self) -> polymetis_pb2.GripperState:
        """Returns the state of the gripper
        Returns:
            gripper state (polymetis_pb2.GripperState)
        """
        return self.grpc_connection.GetState(EMPTY)

    def _get_max_width(self) -> float:
        max_width = getattr(getattr(self, "metadata", None), "max_width", 0.0)
        if not math.isfinite(max_width) or max_width <= 0.0:
            raise RuntimeError("Gripper maximum width is unavailable")
        return max_width

    def get_open_fraction(self) -> float:
        """Return gripper openness from 0 (closed) to 1 (fully open)."""
        width = self.get_state().width
        if not math.isfinite(width):
            raise RuntimeError("Gripper width is unavailable")
        return min(max(width / self._get_max_width(), 0.0), 1.0)

    def goto_open_fraction(
        self, open_fraction: float, speed: float, force: float, blocking: bool = True
    ) -> None:
        """Move to an openness from 0 (closed) to 1 (fully open)."""
        if not math.isfinite(open_fraction) or not 0.0 <= open_fraction <= 1.0:
            raise ValueError("open_fraction must be between 0 and 1")
        self.goto(
            width=open_fraction * self._get_max_width(),
            speed=speed,
            force=force,
            blocking=blocking,
        )

    def goto(
        self, 
        width: float, 
        speed: float, 
        force: float, 
        epsilon_inner: float = -1.0,
        epsilon_outer: float = -1.0,
        blocking: bool = True,):
        """Commands the gripper to a certain width
        Args:
            pos: Target width
            vel: Velocity of the movement
            force: Maximum force the gripper will exert
        """
        cmd = polymetis_pb2.GripperCommand(
            width=width, speed=speed, force=force, grasp=False
        )
        
        cmd.timestamp.GetCurrentTime()

        self._send_gripper_command(self.grpc_connection.Goto, cmd, blocking=blocking)

    def grasp(
        self,
        speed: float,
        force: float,
        grasp_width: float = 0.0,
        epsilon_inner: float = -1.0,
        epsilon_outer: float = -1.0,
        blocking: bool = True,
    ):
        """Commands the gripper to a close

        For Robotiq grippers, this is equivalent to calling `goto` with grasp_width (0 by default).

        For the Franka Hand, see documentation for franka::Gripper::move in libfranka.
        Basically the gripper will perform the grasp if grasp_width - epsilon_inner < final_width < grasp_width + epsilon_outer.
        Else the gripper will not continue to exert force.
        The default width and epsilon values ensures that the gripper closes to the minimum width possible and continues to exert force.


        Args:
            vel: Velocity of the movement
            force: Maximum force the gripper will exert
            grasp_width: Target width of the grasp
            epsilon_inner: Maximum tolerated deviation when the actual grasped width is smaller than the commanded grasp width
            epsilon_outer: Maximum tolerated deviation when the actual grasped width is larger than the commanded grasp width
        """
        cmd = polymetis_pb2.GripperCommand(
            width=grasp_width,
            speed=speed,
            force=force,
            grasp=True,
            epsilon_inner=epsilon_inner,
            epsilon_outer=epsilon_outer,
        )
        cmd.timestamp.GetCurrentTime()

        self._send_gripper_command(self.grpc_connection.Goto, cmd, blocking=blocking)
