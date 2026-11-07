"""Compare recorded absolute targets with their raw robot samples, without hardware."""

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as parquet

from .control.geometry import (
    RECORDED_GRIPPER_MAX_WIDTH_M,
    gripper_fraction_to_width,
    gripper_width_to_fraction,
    measured_state,
    parse_action,
    rotation_six_to_quaternion_xyzw,
)
from .control.robot import SafetyLimits


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data/lerobot_v3/stack_bowl"
DEFAULT_RAW = ROOT / "data/raw/stack_bowl"
REPORT = Path(__file__).resolve().parent / "dataset_check_report.json"


def _values(table, name, width):
    column = table[name].combine_chunks()
    return column.values.to_numpy().reshape(-1, width).astype(np.float64)


def _angles(first, second):
    first = first / np.linalg.norm(first, axis=1, keepdims=True)
    second = second / np.linalg.norm(second, axis=1, keepdims=True)
    dots = np.abs(np.sum(first * second, axis=1))
    return 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))


def _raw_indices(cycles, requested):
    if not np.all(np.diff(cycles) > 0):
        raise ValueError("raw control cycle identifiers must increase")
    indices = np.searchsorted(cycles, requested)
    if np.any(indices >= len(cycles)) or not np.array_equal(cycles[indices], requested):
        raise ValueError("audit references a missing raw control cycle")
    return indices


def _take(dataset, indices):
    unique, inverse = np.unique(indices, return_inverse=True)
    return dataset[unique][inverse]


def _maximum(report, name, values):
    report["maximum_errors"][name] = max(report["maximum_errors"].get(name, 0.0),
                                         float(np.max(values, initial=0.0)))


def check_dataset(dataset=DEFAULT_DATASET, raw=DEFAULT_RAW):
    dataset = Path(dataset)
    raw = Path(raw)
    with (dataset / "meta/info.json").open() as stream:
        info = json.load(stream)
    with (dataset / "meta/conversion.json").open() as stream:
        conversion = json.load(stream)
    if info["fps"] != 30 or conversion["action_representation"] != "absolute_target_eef10":
        raise ValueError("dataset does not contain the expected 30 hertz absolute targets")
    with (dataset / "source_episode_map.csv").open(newline="") as stream:
        episode_map = {int(row["episode_index"]): row["raw_episode_id"] for row in csv.DictReader(stream)}

    audit = parquet.read_table(dataset / "alignment_audit.parquet")
    audit_episode = audit["episode_index"].to_numpy()
    limits = SafetyLimits((-0.85, -0.85, -0.05), (0.85, 0.85, 0.85))
    report = {
        "episodes": 0, "frames": 0, "maximum_errors": {},
        "maximum_distances": {}, "former_update_limit_exceeded": {"position": 0, "rotation": 0},
        "target_distance_limit_exceeded": {"position": 0, "rotation": 0},
    }
    for episode in range(info["total_episodes"]):
        path = dataset / info["data_path"].format(chunk_index=0, file_index=episode)
        table = parquet.read_table(path, columns=["episode_index", "frame_index", "observation.state", "action"])
        state = _values(table, "observation.state", 10)
        action = _values(table, "action", 10)
        frame = table["frame_index"].to_numpy()
        mask = audit_episode == episode
        aligned = audit.filter(mask)
        if (not np.all(table["episode_index"].to_numpy() == episode)
                or not np.array_equal(frame, aligned["frame_index"].to_numpy())):
            raise ValueError("converted rows and alignment rows differ in episode %d" % episode)
        raw_path = raw / episode_map[episode]
        with (raw_path / "metadata.json").open() as stream:
            metadata = json.load(stream)
        if metadata["eef_frame"] != "fr3_link8" or metadata["quaternion_order"] != "xyzw":
            raise ValueError("raw pose convention differs in episode %d" % episode)
        if abs(metadata["gripper_max_width_m"] - RECORDED_GRIPPER_MAX_WIDTH_M) > 1e-9:
            raise ValueError("recorded gripper width differs in episode %d" % episode)
        with h5py.File(raw_path / "robot.hdf5") as recording:
            cycles = recording["timestamps/control_cycle_id"][:]
            observation_indices = _raw_indices(cycles, aligned["observation_cycle_id"].to_numpy())
            gripper_indices = _raw_indices(cycles, aligned["gripper_cycle_id"].to_numpy())
            command_indices = _raw_indices(cycles, aligned["command_cycle_id"].to_numpy())
            observation_pose = _take(recording["observations/eef_pose"], observation_indices)
            observation_six = _take(recording["observations/eef_pose_6d"], observation_indices)
            gripper_width = _take(recording["observations/gripper_state"], gripper_indices)[:, 0]
            target_pose = _take(recording["actions/target_eef_pose"], command_indices)
            target_six = _take(recording["actions/target_eef_pose_6d"], command_indices)
            target_gripper = _take(recording["actions/target_gripper"], command_indices)[:, 0]

        expected_state = np.stack([
            measured_state(pose[:3], pose[3:], gripper_width_to_fraction(width))
            for pose, width in zip(observation_pose, gripper_width)
        ])
        expected_action = np.column_stack((target_six, 2 * np.clip(target_gripper, 0, 1) - 1))
        parsed_actions = [parse_action(row) for row in action]
        parsed_quaternions = np.stack([target.quaternion_xyzw for target in parsed_actions])
        parsed_grippers = np.array([target.gripper_open_fraction for target in parsed_actions])
        parsed_widths = np.array([gripper_fraction_to_width(value) for value in parsed_grippers])
        state_quaternions = np.stack([rotation_six_to_quaternion_xyzw(row[3:9]) for row in state])

        _maximum(report, "state_position_m", np.linalg.norm(state[:, :3] - observation_pose[:, :3], axis=1))
        _maximum(report, "state_rotation_six", np.max(np.abs(state[:, 3:9] - observation_six[:, 3:9]), axis=1))
        _maximum(report, "state_quaternion_rad", _angles(state_quaternions, observation_pose[:, 3:]))
        _maximum(report, "state_gripper", np.abs(state[:, 9] - expected_state[:, 9]))
        _maximum(report, "action_position_m", np.linalg.norm(action[:, :3] - target_pose[:, :3], axis=1))
        _maximum(report, "action_rotation_six", np.max(np.abs(action[:, 3:9] - target_six[:, 3:9]), axis=1))
        _maximum(report, "action_quaternion_rad", _angles(parsed_quaternions, target_pose[:, 3:]))
        _maximum(report, "action_gripper", np.abs(action[:, 9] - expected_action[:, 9]))
        _maximum(report, "parsed_gripper", np.abs(parsed_grippers - np.clip(target_gripper, 0, 1)))
        _maximum(report, "parsed_gripper_width_m", np.abs(
            parsed_widths - np.clip(target_gripper, 0, 1) * metadata["gripper_max_width_m"]))
        _maximum(report, "reconstructed_state", np.max(np.abs(state - expected_state), axis=1))

        measured_distance = np.linalg.norm(action[:, :3] - state[:, :3], axis=1)
        measured_angle = _angles(parsed_quaternions, state_quaternions)
        report["target_distance_limit_exceeded"]["position"] += int(np.count_nonzero(
            measured_distance > limits.maximum_target_distance_m))
        report["target_distance_limit_exceeded"]["rotation"] += int(np.count_nonzero(
            measured_angle > limits.maximum_target_rotation_rad))
        for name, values in (("target_from_state_position_m", measured_distance),
                             ("target_from_state_rotation_rad", measured_angle)):
            report["maximum_distances"][name] = max(report["maximum_distances"].get(name, 0.0),
                                                      float(np.max(values, initial=0.0)))
        step_distance = np.linalg.norm(np.diff(action[:, :3], axis=0), axis=1)
        step_angle = _angles(parsed_quaternions[1:], parsed_quaternions[:-1])
        report["maximum_distances"]["consecutive_target_position_m"] = max(
            report["maximum_distances"].get("consecutive_target_position_m", 0.0),
            float(np.max(step_distance, initial=0.0)))
        report["maximum_distances"]["consecutive_target_rotation_rad"] = max(
            report["maximum_distances"].get("consecutive_target_rotation_rad", 0.0),
            float(np.max(step_angle, initial=0.0)))
        report["former_update_limit_exceeded"]["position"] += int(np.count_nonzero(step_distance > 0.005))
        report["former_update_limit_exceeded"]["rotation"] += int(np.count_nonzero(step_angle > 0.04))
        report["episodes"] += 1
        report["frames"] += len(action)

    if report["frames"] != info["total_frames"]:
        raise ValueError("dataset frame count differs from its metadata")
    tolerance = {
        "state_position_m": 1e-6, "state_rotation_six": 1e-6,
        "state_quaternion_rad": 1e-6, "state_gripper": 1e-6,
        "action_position_m": 1e-6, "action_rotation_six": 1e-6,
        "action_quaternion_rad": 1e-6, "action_gripper": 1e-6,
        "parsed_gripper": 1e-6, "parsed_gripper_width_m": 1e-6,
        "reconstructed_state": 1e-6,
    }
    report["passed"] = all(report["maximum_errors"][name] <= limit for name, limit in tolerance.items())
    report["tolerances"] = tolerance
    return report


def main():
    parser = argparse.ArgumentParser(description="Check converted robot poses against raw recordings")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    arguments = parser.parse_args()
    report = check_dataset(arguments.dataset, arguments.raw)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
