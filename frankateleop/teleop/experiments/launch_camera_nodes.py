from dataclasses import dataclass
from multiprocessing import Process
from typing import Optional

import tyro

from teleop.cameras.realsense_camera import RealSenseCamera, get_device_ids
from teleop.zmq_core.camera_node import ZMQServerCamera


@dataclass
class Args:
    hostname: str = "127.0.0.1"
    wrist_serial: Optional[str] = None
    base_serial: Optional[str] = None
    color_only: bool = False
    # hostname: str = "128.32.175.167"


def launch_server(port: int, camera_id: int, args: Args):
    camera = RealSenseCamera(camera_id, color_only=args.color_only)
    server = ZMQServerCamera(camera, port=port, host=args.hostname)
    print(f"Starting camera server on port {port}")
    server.serve()


def main(args):
    if args.color_only:
        if not args.wrist_serial or not args.base_serial or args.wrist_serial == args.base_serial:
            raise ValueError("RGB collection requires distinct wrist and base serial numbers")
        ids = [args.wrist_serial, args.base_serial]
    else:
        ids = get_device_ids()
    camera_port = 5000
    camera_servers = []
    for camera_id in ids:
        # start a python process for each camera
        print(f"Launching camera {camera_id} on port {camera_port}")
        camera_servers.append(
            Process(target=launch_server, args=(camera_port, camera_id, args))
        )
        camera_port += 1

    for server in camera_servers:
        server.start()


if __name__ == "__main__":
    main(tyro.cli(Args))
