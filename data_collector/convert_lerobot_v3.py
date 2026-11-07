"""Resample timestamped FR3 raw episodes and export LeRobot version 3."""

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np

from fr3_schema import Fr3Kinematics

CAMERAS = ("head_camera", "left_wrist_camera")
VIDEO_KEYS = {slot: "observation.images." + slot for slot in CAMERAS}
URDF = (Path(__file__).resolve().parents[1] /
        "frankateleop/polymetis/polymetis/python/polymetis/data/fr3/fr3.urdf")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nearest(times, target):
    index = bisect.bisect_left(times, target)
    candidates = [i for i in (index - 1, index) if 0 <= i < len(times)]
    return min(candidates, key=lambda i: abs(times[i] - target))


def _nearest_deltas(times, targets):
    positions = np.searchsorted(times, targets)
    before = np.clip(positions - 1, 0, len(times) - 1)
    after = np.clip(positions, 0, len(times) - 1)
    closest = np.where(np.abs(times[before] - targets) <=
                       np.abs(times[after] - targets), times[before], times[after])
    return closest - targets


def _best_phase_offset(first, last, count, period_ns, streams, command_times,
                       max_frame_age_ns, max_state_age_ns, max_command_hold_ns):
    """Minimize the largest sensor offset over one sampling period."""
    offsets = np.rint(np.arange(count) * period_ns).astype(np.int64)
    maximum_shift = min(round(period_ns) - 1, last - first)
    shifts = list(range(0, maximum_shift + 1, 10_000))
    if shifts[-1] != maximum_shift:
        shifts.append(maximum_shift)
    stream_arrays = {name: np.asarray(times, dtype=np.int64)
                     for name, times in streams.items()}
    command_array = np.asarray(command_times, dtype=np.int64)
    best_score = None
    best_shift = 0
    best_count = count
    for shift in shifts:
        candidate_count = int(np.searchsorted(offsets, last - first - shift,
                                              side="right"))
        if candidate_count == 0:
            continue
        targets = first + shift + offsets[:candidate_count]
        command_indices = np.searchsorted(command_array, targets, side="right") - 1
        if np.any(command_indices < 0):
            continue
        maximum_hold = int(np.max(targets - command_array[command_indices]))
        if maximum_hold > max_command_hold_ns:
            continue
        maximum_sensor_offset = 0
        for name, times in stream_arrays.items():
            maximum_offset = int(np.max(np.abs(_nearest_deltas(times, targets))))
            limit = max_frame_age_ns if name in CAMERAS else max_state_age_ns
            if maximum_offset > limit:
                break
            maximum_sensor_offset = max(maximum_sensor_offset, maximum_offset)
        else:
            score = (maximum_sensor_offset, maximum_hold, -candidate_count, shift)
            if best_score is None or score < best_score:
                best_score = score
                best_shift = shift
                best_count = candidate_count
    return best_shift, best_count


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0}
    return {"count": int(len(values)), "minimum": float(values.min()),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "maximum": float(values.max())}


def _read_robot(episode, model):
    import h5py

    with h5py.File(episode / "robot.hdf5") as file:
        require(file.attrs.get("schema_version") == "fr3_raw_v1",
                "Unsupported raw robot schema")
        fields = {name: file[name][:] for name in (
            "timestamps/control_cycle_id", "timestamps/observation_ns",
            "timestamps/command_ns", "timestamps/gripper_observation_ns",
            "observations/joint_positions", "observations/eef_pose",
            "observations/joint_velocities", "observations/eef_pose_6d",
            "observations/gripper_state", "actions/target_joint_positions",
            "actions/target_eef_pose", "actions/target_eef_pose_6d",
            "actions/target_gripper",
            "diagnostics/observation_valid", "diagnostics/command_valid",
            "diagnostics/command_status")}
    count = len(fields["timestamps/control_cycle_id"])
    require(count > 0 and all(len(values) == count for values in fields.values()),
            "Robot fields must have equal nonzero row counts")
    ids = fields["timestamps/control_cycle_id"]
    require(np.all(np.diff(ids) == 1), "Control cycle identifiers must be consecutive")
    observed = fields["diagnostics/observation_valid"].astype(bool)
    commanded = fields["diagnostics/command_valid"].astype(bool)
    status = fields["diagnostics/command_status"]
    require(np.all((status == 1) == commanded), "Command status and validity disagree")
    paired = observed & commanded
    require(np.all(fields["timestamps/command_ns"][paired] >=
                   fields["timestamps/observation_ns"][paired]),
            "A control cycle submitted its command before its observation")
    obs_time = fields["timestamps/observation_ns"][observed]
    grip_time = fields["timestamps/gripper_observation_ns"][observed]
    cmd_time = fields["timestamps/command_ns"][commanded]
    require(len(obs_time) and len(cmd_time) and np.all(np.diff(obs_time) > 0) and
            np.all(np.diff(grip_time) > 0) and np.all(np.diff(cmd_time) > 0),
            "Valid robot timestamps are missing or unordered")
    require(np.all(obs_time > 0) and np.all(grip_time > 0) and
            np.all(cmd_time > 0), "Invalid host timestamps")
    joints = fields["actions/target_joint_positions"][commanded]
    require(np.all(np.isfinite(joints)) and
            np.all(joints >= model.lower - 1e-8) and
            np.all(joints <= model.upper + 1e-8), "Submitted target outside joint limits")
    require(np.isfinite(fields["observations/joint_velocities"][observed]).all(),
            "Measured joint velocity is invalid")
    for kind, mask, joint_key, pose_key, full_pose_key in (
            ("observation", observed, "observations/joint_positions",
             "observations/eef_pose_6d", "observations/eef_pose"),
            ("action", commanded, "actions/target_joint_positions",
             "actions/target_eef_pose_6d", "actions/target_eef_pose")):
        selected = np.flatnonzero(mask)
        for index in selected:
            joint = fields[joint_key][index]
            pose = fields[pose_key][index]
            require(np.isfinite(joint).all() and np.isfinite(pose).all(),
                    f"Non-finite {kind} at cycle {ids[index]}")
            computed_full, computed = model.pose(joint)
            full_pose = fields[full_pose_key][index]
            require(np.allclose(pose, computed, atol=1e-5, rtol=0) and
                    np.allclose(full_pose[:3], computed_full[:3], atol=1e-5, rtol=0) and
                    abs(abs(float(np.dot(full_pose[3:], computed_full[3:]))) - 1) <= 1e-5,
                    f"Forward kinematics mismatch at cycle {ids[index]}")
    return fields, observed, commanded


def _read_camera(episode, slot):
    frames = read_jsonl(episode / slot / "frames.jsonl")
    require(frames and all(row["frame_index"] == index and row["valid"] is True
                           for index, row in enumerate(frames)),
            f"{slot}: invalid or discontinuous frame index")
    times = [int(row["host_timestamp_ns"]) for row in frames]
    require(all(a < b for a, b in zip(times, times[1:])),
            f"{slot}: host timestamps must increase")
    return frames, times


def _encode_video(source, target, indices, expected, fps):
    import cv2

    capture = cv2.VideoCapture(str(source))
    require(capture.isOpened(), f"Cannot read {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    require((width, height) == (640, 480), f"{source}: expected 640 by 480 video")
    target.parent.mkdir(parents=True, exist_ok=True)
    # The container carries one frame per resampled row, including intentional repeats.
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(fps), (width, height))
    require(writer.isOpened(), f"Cannot write {target}")
    cursor = 0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            while cursor < len(indices) and indices[cursor] == frame_index:
                writer.write(frame)
                cursor += 1
            frame_index += 1
    finally:
        writer.release()
        capture.release()
    require(frame_index == expected and cursor == len(indices),
            f"{source}: video and frame index disagree")
    check = cv2.VideoCapture(str(target))
    try:
        count = 0
        while check.read()[0]:
            count += 1
        require(count == len(indices), f"{target}: encoded frame count mismatch")
    finally:
        check.release()


def convert_episode(episode, output, episode_index, fps, task_index,
                    max_frame_age_ns=25_000_000, max_state_age_ns=20_000_000,
                    max_command_hold_ns=50_000_000, optimize_phase=True):
    """Convert one episode, returning its row count and quality audit."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    episode = Path(episode)
    metadata = read_json(episode / "metadata.json")
    require(metadata.get("schema_version") == "fr3_raw_v1", "Unsupported episode schema")
    require(metadata.get("eef_frame") == "fr3_link8", "Unexpected tool frame")
    require(metadata.get("fk_model_sha256") == sha256(URDF), "Robot model differs from capture")
    require(metadata.get("operator_success") is True, "Episode lacks operator success")
    summary = read_json(episode / "summary.json")
    require(summary.get("status") == "raw_unverified", "Raw episode failed capture")
    checksums = read_json(episode / "checksums.json")
    for name, digest in checksums.items():
        require(sha256(episode / name) == digest, f"Raw checksum mismatch: {name}")
    model = Fr3Kinematics(URDF)
    robot, observed, commanded = _read_robot(episode, model)
    obs_indices = np.flatnonzero(observed)
    cmd_indices = np.flatnonzero(commanded)
    obs_times = robot["timestamps/observation_ns"][observed].tolist()
    grip_times = robot["timestamps/gripper_observation_ns"][observed].tolist()
    cmd_times = robot["timestamps/command_ns"][commanded].tolist()
    cameras = {slot: _read_camera(episode, slot) for slot in CAMERAS}
    first = max(obs_times[0], grip_times[0], cmd_times[0],
                *(cameras[slot][1][0] for slot in CAMERAS))
    last = min(obs_times[-1], grip_times[-1], cmd_times[-1] + max_command_hold_ns,
               *(cameras[slot][1][-1] for slot in CAMERAS))
    require(last > first, "Robot and camera streams have no common time interval")
    period_ns = 1_000_000_000 / fps
    count = int(math.floor((last - first) / period_ns)) + 1
    require(count > 0, "No resampled frames")
    original_count = count
    phase_offset_ns = 0
    if optimize_phase:
        streams = {"state": obs_times, "gripper": grip_times,
                   **{slot: cameras[slot][1] for slot in CAMERAS}}
        phase_offset_ns, count = _best_phase_offset(
            first, last, count, period_ns, streams, cmd_times,
            max_frame_age_ns, max_state_age_ns, max_command_hold_ns)
    rows = []
    audit = []
    chosen = {slot: [] for slot in CAMERAS}
    max_width = float(metadata["gripper_max_width_m"])
    require(max_width > 0, "Invalid gripper width calibration")
    for frame_index in range(count):
        target_ns = first + phase_offset_ns + round(frame_index * period_ns)
        oi = _nearest(obs_times, target_ns)
        gi = _nearest(grip_times, target_ns)
        ci = bisect.bisect_right(cmd_times, target_ns) - 1
        require(ci >= 0 and abs(obs_times[oi] - target_ns) <= max_state_age_ns and
                abs(grip_times[gi] - target_ns) <= max_state_age_ns and
                target_ns - cmd_times[ci] <= max_command_hold_ns,
                f"Robot state or held command is too old at frame {frame_index}")
        observation_row = int(obs_indices[oi])
        gripper_row = int(obs_indices[gi])
        command_row = int(cmd_indices[ci])
        obs_pose = robot["observations/eef_pose_6d"][observation_row]
        width = float(robot["observations/gripper_state"][gripper_row, 0])
        fraction = float(robot["actions/target_gripper"][command_row, 0])
        require(np.isfinite(width) and 0 <= width <= max_width * 1.05 and
                np.isfinite(fraction) and 0 <= fraction <= 1,
                f"Invalid gripper value at frame {frame_index}")
        target_joint = robot["actions/target_joint_positions"][command_row]
        _, action_pose = model.pose(target_joint)
        state = np.r_[obs_pose, np.clip(2 * width / max_width - 1, -1, 1)]
        action = np.r_[action_pose, 2 * fraction - 1]
        row = {"episode_index": episode_index, "frame_index": frame_index,
               "timestamp": frame_index / fps, "task_index": task_index,
               "index": None, "observation.state": state.astype(np.float32).tolist(),
               "action": action.astype(np.float32).tolist()}
        trace = {"episode_index": episode_index, "frame_index": frame_index,
                 "target_host_ns": target_ns,
                 "observation_cycle_id": int(robot["timestamps/control_cycle_id"][observation_row]),
                 "gripper_cycle_id": int(robot["timestamps/control_cycle_id"][gripper_row]),
                 "command_cycle_id": int(robot["timestamps/control_cycle_id"][command_row]),
                 "observation_delta_ns": obs_times[oi] - target_ns,
                 "gripper_delta_ns": grip_times[gi] - target_ns,
                 "command_age_ns": target_ns - cmd_times[ci]}
        for slot in CAMERAS:
            frames, times = cameras[slot]
            fi = _nearest(times, target_ns)
            require(abs(times[fi] - target_ns) <= max_frame_age_ns,
                    f"{slot}: no nearby frame at resampled row {frame_index}")
            chosen[slot].append(fi)
            trace[slot + "_frame_index"] = fi
            trace[slot + "_delta_ns"] = times[fi] - target_ns
        rows.append(row)
        audit.append(trace)
    for field in ("observation.state", "action"):
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        require(np.isfinite(values).all(), f"Non-finite {field}")
        require(np.max(np.abs(np.diff(values[:, :3], axis=0)), initial=0) <= 0.25,
                f"Abnormal end-effector jump in {field}")
    exported_joints = robot["actions/target_joint_positions"][
        [int(cmd_indices[bisect.bisect_right(cmd_times, item["target_host_ns"]) - 1])
         for item in audit]]
    maximum_joint_jump = float(np.max(np.abs(np.diff(exported_joints, axis=0)), initial=0))
    require(maximum_joint_jump <= 1.0, "Abnormal target joint jump after resampling")
    data_path = output / "data" / "chunk-000" / f"file-{episode_index:03d}.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    for field in ("action", "observation.state"):
        table = table.set_column(table.schema.get_field_index(field), field,
                                 pa.array([row[field] for row in rows],
                                          type=pa.list_(pa.float32(), 10)))
    pq.write_table(table, data_path)
    require(pq.read_table(data_path).num_rows == count, "Exported robot table cannot be read")
    for slot in CAMERAS:
        path = output / "videos" / VIDEO_KEYS[slot] / "chunk-000" / f"file-{episode_index:03d}.mp4"
        _encode_video(episode / slot / "color.mp4", path, chosen[slot],
                      len(cameras[slot][0]), fps)
    paired = observed & commanded
    cycle_delta = (robot["timestamps/command_ns"][paired] -
                   robot["timestamps/observation_ns"][paired]) / 1e6
    quality = {
        "raw_control_cycles": int(len(robot["timestamps/control_cycle_id"])),
        "missing_observations": int(np.count_nonzero(~observed)),
        "missing_commands": int(np.count_nonzero(~commanded)),
        "command_status_counts": {str(code): int(np.count_nonzero(
            robot["diagnostics/command_status"] == code)) for code in range(4)},
        "paired_observation_command_delta_ms": _distribution(cycle_delta),
        "observation_period_ms": _distribution(np.diff(obs_times) / 1e6),
        "command_period_ms": _distribution(np.diff(cmd_times) / 1e6),
        "resampled_frames": count,
        "phase_offset_ms": phase_offset_ns / 1e6,
        "phase_discarded_frames": original_count - count,
        "resampled_observation_delta_ms": _distribution(
            [item["observation_delta_ns"] / 1e6 for item in audit]),
        "resampled_gripper_delta_ms": _distribution(
            [item["gripper_delta_ns"] / 1e6 for item in audit]),
        "resampled_command_hold_ms": _distribution(
            [item["command_age_ns"] / 1e6 for item in audit]),
        "maximum_target_joint_jump_rad": maximum_joint_jump,
        "camera_alignment_ms": {slot: _distribution(
            [item[slot + "_delta_ns"] / 1e6 for item in audit]) for slot in CAMERAS},
        "camera_frame_repeats": {slot: sum(a == b for a, b in
            zip(chosen[slot], chosen[slot][1:])) for slot in CAMERAS},
        "camera_device_drops": metadata.get("camera_drops", {}),
        "forward_kinematics_consistent": True,
        "joint_limits_valid": True,
        "interpolation_method": "nearest_valid_observation",
        "action_method": "zero_order_hold",
    }
    quality_path = output / "quality" / f"episode-{episode_index:03d}.json"
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.write_text(json.dumps(quality, indent=2) + "\n")
    return count, audit


def verify_openwam(output, split):
    """Read a sample with the configured FR3 dual-camera dataset reader."""
    import sys
    import types

    root = Path(__file__).resolve().parents[1] / "OpenWAM"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # The package initializer eagerly imports unrelated dataset families and
    # their dependencies. Load the real FR3 reader package directly.
    import openwam
    package = types.ModuleType("openwam.dataloader")
    package.__path__ = [str(root / "openwam" / "dataloader")]
    sys.modules["openwam.dataloader"] = package
    from openwam.dataloader.put_block_in_the_bowl import PutBlockInTheBowlDataset

    dataset = PutBlockInTheBowlDataset(str(output), split=split, multiview=True,
                                       normalize_mode="none", num_frames=2)
    require(len(dataset) > 0, "OpenWAM reader found no frames")
    sample = dataset[0]
    require(np.asarray(sample["action"]).shape == (1, 10) and
            np.asarray(sample["proprio"]).shape == (1, 10) and
            len(sample["video"]) > 0 and dataset._wrist_fail_count == 0,
            "OpenWAM did not parse state, action and both cameras")
    return True


def export_candidates(plan):
    """Select saved operator-successful episodes without an acceptance step."""
    return [row for row in plan
            if (row.get("success") or "").strip().lower() == "true"
            and (row.get("raw_episode_id") or "").strip()
            and (row.get("raw_path") or "").strip()]


def main():
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--preview-episodes", type=int)
    parser.add_argument("--optimize-phase", action="store_true", default=True,
                        help="Optimize timestamp alignment (enabled by default)")
    parser.add_argument("--max-frame-age-ms", type=float, default=25)
    parser.add_argument("--max-state-age-ms", type=float, default=20)
    parser.add_argument("--max-command-hold-ms", type=float, default=50)
    parser.add_argument("--openwam-python", type=Path,
                        help="Python interpreter with OpenWAM video dependencies")
    args = parser.parse_args()
    require(0 < args.fps <= 30 and args.max_frame_age_ms > 0 and
            args.max_state_age_ms > 0 and args.max_command_hold_ms > 0,
            "Invalid export sampling or age limit")
    require(not args.output.exists(), "Output already exists")
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        plan = list(csv.DictReader(handle))
    selected = export_candidates(plan)
    if args.preview_episodes is not None:
        require(args.preview_episodes > 0, "Preview episode count must be positive")
        selected = selected[:args.preview_episodes]
    require(selected, "No eligible episodes in manifest")
    require(len({row["task_id"] for row in selected}) == 1 and
            len({row["raw_path"] for row in selected}) == len(selected),
            "Export requires one task with distinct raw episodes")
    selected.sort(key=lambda row: (row["split"] != "train", int(row["planned_order"])))
    prompts = sorted({row["prompt"] for row in selected})
    tasks = {prompt: index for index, prompt in enumerate(prompts)}
    partial = args.output.with_name(args.output.name + f".partial-{os.getpid()}")
    require(not partial.exists(), "Partial export already exists")
    partial.mkdir(parents=True)
    episodes = []
    audit = []
    mapping = []
    cursor = 0
    try:
        for index, item in enumerate(selected):
            source = Path(item["raw_path"])
            require(source.name == item["raw_episode_id"] and source.is_dir(),
                    "Manifest raw episode path mismatch")
            metadata = read_json(source / "metadata.json")
            require(metadata["task_id"] == item["task_id"] and
                    metadata["planned_id"] == item["planned_episode_id"] and
                    metadata["prompt"] == item["prompt"],
                    "Manifest and episode metadata disagree")
            length, trace = convert_episode(
                source, partial, index, args.fps, tasks[item["prompt"]],
                round(args.max_frame_age_ms * 1e6),
                round(args.max_state_age_ms * 1e6),
                round(args.max_command_hold_ms * 1e6), args.optimize_phase)
            path = partial / "data" / "chunk-000" / f"file-{index:03d}.parquet"
            table = pq.read_table(path)
            table = table.set_column(table.schema.get_field_index("index"), "index",
                                     pa.array(range(cursor, cursor + length), type=pa.int64()))
            pq.write_table(table, path)
            episode_row = {"episode_index": index, "length": length,
                           "tasks": [item["prompt"]], "data/chunk_index": 0,
                           "data/file_index": index, "dataset_from_index": cursor,
                           "dataset_to_index": cursor + length}
            for slot in CAMERAS:
                prefix = "videos/" + VIDEO_KEYS[slot]
                episode_row.update({prefix + "/chunk_index": 0, prefix + "/file_index": index,
                                    prefix + "/from_timestamp": 0.0,
                                    prefix + "/to_timestamp": length / args.fps})
            episodes.append(episode_row)
            audit.extend(trace)
            mapping.append({"episode_index": index, "raw_episode_id": source.name,
                            "planned_episode_id": item["planned_episode_id"],
                            "split": item["split"]})
            cursor += length
        meta = partial / "meta"
        (meta / "episodes" / "chunk-000").mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(episodes),
                       meta / "episodes" / "chunk-000" / "file-000.parquet")
        pd.DataFrame({"task_index": range(len(prompts))},
                     index=pd.Index(prompts, name="task")).to_parquet(meta / "tasks.parquet")
        pq.write_table(pa.Table.from_pylist(audit), partial / "alignment_audit.parquet")
        features = {"action": {"dtype": "float32", "shape": [10]},
                    "observation.state": {"dtype": "float32", "shape": [10]},
                    "episode_index": {"dtype": "int64", "shape": [1]},
                    "frame_index": {"dtype": "int64", "shape": [1]},
                    "timestamp": {"dtype": "float32", "shape": [1]},
                    "task_index": {"dtype": "int64", "shape": [1]},
                    "index": {"dtype": "int64", "shape": [1]}}
        for slot in CAMERAS:
            features[VIDEO_KEYS[slot]] = {
                "dtype": "video", "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
                "video_info": {"video.fps": args.fps, "video.codec": "mp4v"}}
        train = sum(row["split"] == "train" for row in selected)
        info = {"codebase_version": "v3.0", "robot_type": "fr3", "fps": args.fps,
                "total_episodes": len(episodes), "total_frames": cursor,
                "total_tasks": len(prompts), "total_videos": len(episodes) * 2,
                "total_chunks": 1, "chunks_size": 1000,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": features,
                "splits": {"train": f"0:{train}", "val": f"{train}:{len(selected)}"}}
        (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n")
        stats = {}
        for field in ("action", "observation.state"):
            values = np.concatenate([
                np.asarray(pq.read_table(partial / "data" / "chunk-000" /
                                         f"file-{index:03d}.parquet",
                                         columns=[field])[field].to_pylist(),
                           dtype=np.float64)
                for index in range(train or len(selected))])
            stats[field] = {
                "mean": values.mean(axis=0).tolist(),
                "std": values.std(axis=0).tolist(),
                "min": values.min(axis=0).tolist(),
                "max": values.max(axis=0).tolist(),
                "count": int(len(values))}
        (meta / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        (meta / "conversion.json").write_text(json.dumps({
            "action_representation": "absolute_target_eef10",
            "state_representation": "fr3_link8_eef10",
            "action_frame": "fr3_link8", "state_frame": "fr3_link8",
            "gripper_convention": "minus1_closed_plus1_open",
            "action_time_semantics": "zero_order_hold_from_interface_acknowledgement",
            "state_time_semantics": "nearest_valid_host_observation",
            "camera_time_semantics": "nearest_valid_host_frame",
            "preview_export": args.preview_episodes is not None,
            "phase_optimized": args.optimize_phase,
            "max_frame_age_ms": args.max_frame_age_ms,
            "max_state_age_ms": args.max_state_age_ms,
            "max_command_hold_ms": args.max_command_hold_ms}, indent=2) + "\n")
        with (partial / "source_episode_map.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=mapping[0])
            writer.writeheader()
            writer.writerows(mapping)
        # The same code path reads the emitted table and both videos above.
        # OpenWAM verification runs when its runtime dependencies are available.
        verification = {"parquet_and_videos_readable": True}
        split = "train" if train else "val"
        if args.openwam_python:
            command = ("from pathlib import Path; from convert_lerobot_v3 import verify_openwam; "
                       "import sys; verify_openwam(Path(sys.argv[1]), sys.argv[2])")
            reader_env = os.environ.copy()
            local_dependencies = str(Path(__file__).resolve().parents[1] / ".deps")
            reader_paths = [path for path in reader_env.get("PYTHONPATH", "").split(os.pathsep)
                            if path and os.path.realpath(path) != os.path.realpath(local_dependencies)]
            if reader_paths:
                reader_env["PYTHONPATH"] = os.pathsep.join(reader_paths)
            else:
                reader_env.pop("PYTHONPATH", None)
            subprocess.run([str(args.openwam_python), "-c", command, str(partial), split],
                           cwd=Path(__file__).resolve().parent, env=reader_env, check=True)
            verification["openwam"] = True
        else:
            try:
                verification["openwam"] = verify_openwam(partial, split)
            except ImportError as exc:
                verification["openwam"] = "not_run: " + str(exc)
        (partial / "quality_report.json").write_text(json.dumps(verification, indent=2) + "\n")
        os.replace(partial, args.output)
    except BaseException:
        # Keep the partial export for diagnosis; it is never a completed dataset.
        raise
    print(f"Converted {len(episodes)} episodes and {cursor} frames to {args.output}")


if __name__ == "__main__":
    main()
