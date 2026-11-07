import os
import subprocess

current_file_path = os.path.abspath(__file__)


def run_docker_container():
    user = os.getenv("USER")
    container_name = f"teleop_{user}"
    teleop_path = os.path.abspath(os.path.join(current_file_path, "../../"))
    volume_mapping = f"{teleop_path}:/teleop"

    cmd = [
        "docker",
        "run",
        "--runtime=nvidia",
        "--rm",
        "--name",
        container_name,
        "--privileged",
        "--volume",
        volume_mapping,
        "--volume",
        "/home/teleop:/homefolder",
        "--net=host",
        "--volume",
        "/dev/serial/by-id/:/dev/serial/by-id/",
        "-it",
        "teleop:latest",
        "bash",
        "-c",
        "pip install -e third_party/DynamixelSDK/python && exec bash",
    ]

    subprocess.run(cmd)


if __name__ == "__main__":
    run_docker_container()
