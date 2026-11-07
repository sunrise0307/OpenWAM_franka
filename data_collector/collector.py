"""Record one FR3 raw episode with paired robot cycles and two color cameras."""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

from camera_capture import ColorRecorder, SessionCamera


FR3_URDF_PATH = (Path(__file__).resolve().parents[1] /
                "frankateleop/polymetis/polymetis/python/polymetis/data/fr3/fr3.urdf")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_args(args):
    for value, name in ((args.task_id, "task-id"), (args.planned_id, "planned-id")):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
            raise ValueError(f"Invalid {name}")
    if args.episode_id is not None and not re.fullmatch(
            r"episode_[0-9]{6}", args.episode_id):
        raise ValueError("episode-id must match episode_000000")
    if not args.head_serial or not args.wrist_serial or args.head_serial == args.wrist_serial:
        raise ValueError("Two different camera serials are required")
    if not args.instruction or not args.success_criterion:
        raise ValueError("Instruction and success criterion are required")
    if not math.isfinite(args.hz) or not 0 < args.hz <= 100:
        raise ValueError("Control rate must be in (0, 100] Hz")
    if args.duration_s is not None and (not math.isfinite(args.duration_s) or
                                        args.duration_s <= 0):
        raise ValueError("Duration must be positive")
    if not Path(args.urdf).is_file() or not Path(args.plan_manifest).is_file():
        raise ValueError("Robot model or plan manifest is missing")
    if not Path(args.gello_port).exists():
        raise ValueError("GELLO serial port is unavailable")


def planned_row(args):
    with open(args.plan_manifest, newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["task_id"] == args.task_id and
                row["planned_episode_id"] == args.planned_id]
    if len(rows) != 1 or rows[0]["split"] != args.planned_split:
        raise ValueError("Planned task, episode and split must match one manifest row")
    return rows[0]


def connect_grpc(host, arm_port, gripper_port, urdf, ee_link):
    import grpc
    import polymetis_pb2
    import polymetis_pb2_grpc

    arm_channel = grpc.insecure_channel(f"{host}:{arm_port}")
    grip_channel = grpc.insecure_channel(f"{host}:{gripper_port}")
    try:
        grpc.channel_ready_future(arm_channel).result(timeout=5)
        grpc.channel_ready_future(grip_channel).result(timeout=5)
        arm = polymetis_pb2_grpc.PolymetisControllerServerStub(arm_channel)
        grip = polymetis_pb2_grpc.GripperServerStub(grip_channel)
        arm_metadata = arm.GetRobotClientMetadata(polymetis_pb2.Empty(), timeout=3)
        grip_metadata = grip.GetRobotClientMetadata(polymetis_pb2.Empty(), timeout=3)
        if (arm_metadata.urdf_file != Path(urdf).read_text() or
                arm_metadata.ee_link_name != ee_link or arm_metadata.dof != 7 or
                grip_metadata.max_width <= 0):
            raise RuntimeError("Robot server metadata differs from the selected FR3 model")
        return arm_channel, grip_channel, arm, grip, arm_metadata, grip_metadata
    except Exception:
        arm_channel.close()
        grip_channel.close()
        raise


def run(args):
    from continuous_teleop import ContinuousTeleop

    validate_args(args)
    plan = planned_row(args)
    if plan.get("success", "").lower() == "true" and plan.get("raw_episode_id"):
        raise ValueError("The planned episode already has a saved successful recording")
    args.teleop_hz = args.hz
    engine = ContinuousTeleop(args, args.plan_manifest)
    raw_root = Path(args.output_dir).expanduser().resolve() / "raw" / args.task_id
    existing = [int(path.name[8:]) for path in raw_root.glob("episode_*")
                if path.is_dir() and path.name[8:].isdigit()]
    episode_id = args.episode_id or f"episode_{max(existing, default=-1) + 1:06d}"
    try:
        engine.start()
        episode = engine.begin(plan, episode_id)
        try:
            end = None if args.duration_s is None else time.monotonic() + args.duration_s
            while end is None or time.monotonic() < end:
                engine.check()
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        engine.active["metadata"]["operator_success"] = args.observed_result == "success"
        engine.active["metadata"]["observed_result"] = args.observed_result
        engine.finish()
        from task_plan import read_manifest, write_manifest_atomic

        fields, rows = read_manifest(Path(args.plan_manifest))
        for row in rows:
            if row["task_id"] == args.task_id and row["planned_episode_id"] == args.planned_id:
                row.update(raw_episode_id=episode.name, raw_path=str(episode),
                           session_id=args.session_id, operator_id=args.operator,
                           third_person_camera_serial=args.head_serial,
                           wrist_camera_serial=args.wrist_serial, prompt=args.instruction,
                           success=str(args.observed_result == "success").lower(),
                           acceptance_status="pending", rejection_reason="",
                           accepted_episode_id="")
                break
        write_manifest_atomic(Path(args.plan_manifest), fields, rows)
        write_manifest_atomic(episode.parent / "manifest.csv", fields, rows)
        print(episode)
        return episode
    except BaseException:
        if engine.active is not None:
            try:
                engine.finish()
            except Exception:
                pass
        raise
    finally:
        engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--planned-id", required=True)
    parser.add_argument("--episode-id")
    parser.add_argument("--planned-split", choices=("train", "val"), required=True)
    parser.add_argument("--plan-manifest", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--success-criterion", required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--observed-result", choices=("success", "failure", "unknown"),
                        default="unknown")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--head-serial", required=True)
    parser.add_argument("--wrist-serial", required=True)
    parser.add_argument("--gello-port", required=True)
    parser.add_argument("--calibration-dir")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--arm-port", type=int, default=50051)
    parser.add_argument("--gripper-port", type=int, default=50052)
    parser.add_argument("--teleop-port", type=int, default=6001)
    parser.add_argument("--urdf", default=str(FR3_URDF_PATH))
    parser.add_argument("--ee-link", default="fr3_link8")
    parser.add_argument("--hz", type=float, default=100.0)
    parser.add_argument("--duration-s", type=float)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
