from dataclasses import dataclass
from pathlib import Path

import tyro
import torch

from teleop.robots.robot import BimanualRobot, PrintRobot
from teleop.zmq_core.robot_node import ZMQServerRobot


@dataclass
class Args:
    robot: str = "xarm"
    tele_port: int = 6001
    hostname: str = "127.0.0.1"
    robot_ip: str = "127.0.0.1"
    robot_port: int = 50051
    gripper_port: int = 50052


def launch_robot_server(args: Args):
    port = args.tele_port
    if args.robot == "sim_ur":
        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "universal_robots_ur5e" / "ur5e.xml"
        gripper_xml = MENAGERIE_ROOT / "robotiq_2f85" / "2f85.xml"
        from teleop.robots.sim_robot import MujocoRobotServer

        server = MujocoRobotServer(
            xml_path=xml, gripper_xml_path=gripper_xml, port=port, host=args.hostname
        )
        server.serve()
    elif args.robot == "sim_fr3":
        from teleop.robots.sim_robot import MujocoRobotServer

        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "franka_emika_fr3" / "fr3.xml"
        gripper_xml = None
        server = MujocoRobotServer(
            xml_path=xml, gripper_xml_path=gripper_xml, port=port, host=args.hostname
        )
        server.serve()
    elif args.robot == "sim_xarm":
        from teleop.robots.sim_robot import MujocoRobotServer

        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "ufactory_xarm7" / "xarm7.xml"
        gripper_xml = None
        server = MujocoRobotServer(
            xml_path=xml, gripper_xml_path=gripper_xml, port=port, host=args.hostname
        )
        server.serve()

    else:
        fr3_joint_configs = {
            "fr3_left": torch.Tensor([0.580828,0.00354707,0.121271,-1.59745,-1.53366,1.61208,-0.865943]), 
            "fr3_right": torch.Tensor([-0.506681,0.116912,-0.303241,-1.56214,1.69048,1.57112,-0.804331])  
        }
        if args.robot == "xarm":
            from teleop.robots.xarm_robot import XArmRobot

            robot = XArmRobot(ip=args.robot_ip)
        elif args.robot == "ur":
            from teleop.robots.ur import URRobot
            
        elif args.robot == "fr3_left":
            from teleop.robots.fr3 import fr3Robot

            robot = fr3Robot(
                robot_ip=args.robot_ip, 
                franka_port=args.robot_port, 
                frankahand_port=args.gripper_port, 
                joint_positions_desired=fr3_joint_configs["fr3_left"]
                )
        
        elif args.robot == "fr3_right":
            from teleop.robots.fr3 import fr3Robot

            robot = fr3Robot(
                robot_ip=args.robot_ip, 
                franka_port=args.robot_port, 
                frankahand_port=args.gripper_port, 
                joint_positions_desired=fr3_joint_configs["fr3_right"]
                )
       
        elif args.robot == "fr3":
            from teleop.robots.fr3 import fr3Robot

            robot = fr3Robot(
                robot_ip=args.robot_ip, 
                franka_port=args.robot_port, 
                frankahand_port=args.gripper_port
                )
        elif args.robot == "bimanual_ur":
            from teleop.robots.ur import URRobot

            # IP for the bimanual robot setup is hardcoded
            _robot_l = URRobot(robot_ip="192.168.2.10")
            _robot_r = URRobot(robot_ip="192.168.1.10")
            robot = BimanualRobot(_robot_l, _robot_r)
        elif args.robot == "none" or args.robot == "print":
            robot = PrintRobot(8)

        else:
            raise NotImplementedError(
                f"Robot {args.robot} not implemented, choose one of: sim_ur, xarm, ur, bimanual_ur, none"
            )
        server = ZMQServerRobot(robot, port=port, host=args.hostname)
        print(f"Starting robot server on port {port}")
        server.serve()


def main(args):
    launch_robot_server(args)


if __name__ == "__main__":
    main(tyro.cli(Args))

