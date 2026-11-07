"""Keyboard-driven, resumable collection of one task's raw episodes."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import select
import shutil
import subprocess
import sys
import tempfile
import termios
import time
import tty
import uuid
from contextlib import contextmanager
from pathlib import Path

from task_plan import TRAIN_SIZE, VAL_SIZE, ensure_manifest
from continuous_teleop import ContinuousTeleop


SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE = SCRIPT_DIR.parent / "teleop_plan/episode_manifest_template.csv"


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_csv(path, fields, rows):
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".manifest-",
                                     newline="", encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink()
            raise
    os.replace(temporary, path)


def episode_is_complete(row):
    return (
        row.get("success", "").strip().lower() == "true"
        and bool(row.get("raw_episode_id", "").strip())
    )


class CollectionSession:
    def __init__(self, args, engine_factory=ContinuousTeleop):
        self.args = args
        self.output_dir = Path(args.output_dir).expanduser().resolve()
        self.raw_root = self.output_dir / "raw" / args.task_id
        self.manifest = self.output_dir / "plans" / (args.task_id + "_manifest.csv")
        self.engine_factory = engine_factory
        self.alignment_script = SCRIPT_DIR / "align_gello.py"
        self.train_episodes = getattr(args, "train_episodes", TRAIN_SIZE)
        self.val_episodes = getattr(args, "val_episodes", VAL_SIZE)
        self.fields = None
        self.rows = None

    def load_plan(self):
        self.fields, self.rows = ensure_manifest(
            TEMPLATE, self.manifest, self.args.task_id,
            self.train_episodes, self.val_episodes)
        self.raw_root.mkdir(parents=True, exist_ok=True)
        dataset_metadata = self.raw_root / "dataset_metadata.json"
        if not dataset_metadata.exists():
            atomic_json(dataset_metadata, {
                "schema_version": "fr3_raw_v1", "task_id": self.args.task_id,
                "prompt": self.args.instruction, "control_hz_target": self.args.teleop_hz,
                "camera_fps_target": 30, "capture_module": "continuous_teleop"})
        if not (self.raw_root / "manifest.csv").exists():
            atomic_csv(self.raw_root / "manifest.csv", self.fields, self.rows)
        for row in self.rows:
            if episode_is_complete(row):
                episode = self.raw_root / row["raw_episode_id"]

                if not self.valid_episode(episode, row):
                    raise ValueError(
                        "Completed plan row has missing or invalid raw data: " +
                        row["planned_episode_id"]
                    )
        self.recover_successes()

    def valid_episode(self, episode, row):
        try:
            metadata = read_json(episode / "metadata.json")
            summary = read_json(episode / "summary.json")
            return (metadata["task_id"] == self.args.task_id and
                    metadata["planned_id"] == row["planned_episode_id"] and
                    summary["status"] == "raw_unverified" and
                    summary["control_commands"] > 0 and
                    (episode / "checksums.json").is_file() and
                    all((episode / name).is_file() and (episode / name).stat().st_size > 0
                        for name in ("robot.hdf5",
                                     "head_camera/color.mp4", "head_camera/frames.jsonl",
                                     "left_wrist_camera/color.mp4",
                                     "left_wrist_camera/frames.jsonl")))
        except (OSError, KeyError, ValueError, TypeError):
            return False

    def recover_successes(self):
        if not self.raw_root.exists():
            return
        candidates = {}
        for episode in self.raw_root.iterdir():
            if not episode.is_dir() or not (episode / "metadata.json").is_file():
                continue
            try:
                metadata = read_json(episode / "metadata.json")
            except (OSError, ValueError):
                continue
            if (metadata.get("task_id") == self.args.task_id and
                    metadata.get("observed_result") == "success" and
                    metadata.get("operator_success_key") == "t"):
                candidates.setdefault(metadata.get("planned_id"), []).append(episode)
        for row in self.rows:
            if episode_is_complete(row):
                continue
            episodes = [episode for episode in candidates.get(row["planned_episode_id"], [])
                        if self.valid_episode(episode, row)]
            if len(episodes) > 1:
                raise ValueError("Multiple successful raw episodes for " +
                                 row["planned_episode_id"])
            if episodes:
                self.record_success(row, episodes[0], recovering=True)
                print("Recovered saved episode for " + row["planned_episode_id"])

    def next_row(self):
        return next((row for row in self.rows if not episode_is_complete(row)), None)

    def discard(self, episode):
        if episode is None:
            return
        if episode.parent.resolve() != self.raw_root.resolve() or not episode.is_dir():
            raise ValueError("Refusing to remove an unexpected episode path")
        shutil.rmtree(episode)
        print("Discarded " + str(episode))

    def record_success(self, row, episode, recovering=False):
        if not self.valid_episode(episode, row):
            raise ValueError("Collector output is incomplete; plan remains pending")
        metadata_path = episode / "metadata.json"
        metadata = read_json(metadata_path)
        if (metadata.get("observed_result") != "success" or
                metadata.get("operator_success_key") != "t"):
            metadata["observed_result"] = "success"
            metadata["operator_success"] = True
            metadata["operator_success_key"] = "t"
            atomic_json(metadata_path, metadata)
        checksums_path = episode / "checksums.json"
        checksums = read_json(checksums_path)
        checksums["metadata.json"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        atomic_json(checksums_path, checksums)

        updated = dict(row)
        updated.update(raw_episode_id=episode.name, raw_path=str(episode),
                       session_id=metadata["session_id"], operator_id=metadata["operator"],
                       third_person_camera_serial=metadata["cameras"]["head_camera"]["serial"],
                       wrist_camera_serial=metadata["cameras"]["left_wrist_camera"]["serial"],
                       prompt=metadata["instruction"], success="true",
                       acceptance_status="pending", rejection_reason="",
                       accepted_episode_id="")
        index = self.rows.index(row)
        replacement = list(self.rows)
        replacement[index] = updated
        atomic_csv(self.manifest, self.fields, replacement)
        atomic_csv(self.raw_root / "manifest.csv", self.fields, replacement)
        row.update(updated)
        if not recovering:
            print("Saved " + row["planned_episode_id"] + " -> " + str(episode))

    def capture(self, engine, row, read_key):
        self.raw_root.mkdir(parents=True, exist_ok=True)
        existing = [int(path.name[8:]) for path in self.raw_root.glob("episode_*")
                    if path.is_dir() and path.name[8:].isdigit()]
        episode_id = f"episode_{max(existing, default=-1) + 1:06d}"
        episode = engine.begin(row, episode_id)
        print("Recording: t = save, i = discard; teleoperation stays active", flush=True)
        try:
            while True:
                engine.check()
                key = read_key(0.1)
                if key in ("t", "i"):
                    break
        except BaseException:
            try:
                engine.finish(discard=False)
            except Exception:
                pass
            raise
        if key == "i":
            engine.finish(discard=True)
            print("Plan remains pending: " + row["planned_episode_id"])
        else:
            engine.finish()
            try:
                self.record_success(row, episode)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                print("Could not save plan result; raw data retained for " +
                      "inspection: " + str(exc), file=sys.stderr)
                print("Plan remains pending: " + row["planned_episode_id"])

    def run(self, read_key):
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        with (self.manifest.parent / (self.args.task_id + ".lock")).open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another collection session is active for this task")
            self.load_plan()
            plan_size = len(self.rows)
            if self.next_row() is None:
                print(f"{plan_size}/{plan_size} operator-successful raw episodes recorded for " +
                      self.args.task_id + ". Data can be exported directly.")
                return
            result = subprocess.run([sys.executable, str(self.alignment_script),
                                     "--port", self.args.gello_port], check=False)
            if result.returncode != 0:
                raise RuntimeError("GELLO alignment failed; no episode was started")
            engine = self.engine_factory(self.args, self.manifest)
            try:
                engine.start()
                self._run_keys(engine, read_key)
            finally:
                engine.close()

    def _run_keys(self, engine, read_key):
        plan_size = len(self.rows)
        while True:
            engine.check()
            row = self.next_row()
            done = sum(episode_is_complete(item) for item in self.rows)
            if row is None:
                print(f"{plan_size}/{plan_size} operator-successful raw episodes recorded for " +
                      self.args.task_id + ". Data can be exported directly.")
                return
            print("{}/{} | {} | {} | s = start, q = quit".format(
                done, plan_size, row["planned_episode_id"], row["split"]), flush=True)
            while True:
                engine.check()
                key = read_key(0.1)
                if key == "s":
                    self.capture(engine, row, read_key)
                    break
                if key == "q":
                    return


@contextmanager
def terminal_keys():
    descriptor = sys.stdin.fileno()
    old_settings = termios.tcgetattr(descriptor)
    tty.setcbreak(descriptor)
    try:
        def read_key(timeout):
            ready, _, _ = select.select([descriptor], [], [], timeout)
            if not ready:
                return None
            data = os.read(descriptor, 1)
            if not data:
                raise EOFError("Terminal input closed")
            return data.decode("ascii", errors="ignore").lower()
        yield read_key
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, old_settings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--success-criterion", required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--head-serial", required=True)
    parser.add_argument("--wrist-serial", required=True)
    parser.add_argument("--gello-port", required=True)
    parser.add_argument("--calibration-dir")
    parser.add_argument("--train-episodes", type=int, default=TRAIN_SIZE)
    parser.add_argument("--val-episodes", type=int, default=VAL_SIZE)
    parser.add_argument("--teleop-hz", type=float, default=100.0)
    args = parser.parse_args()
    if args.train_episodes < 1 or args.val_episodes < 1:
        parser.error("Training and validation episode counts must be positive")
    if not math.isfinite(args.teleop_hz) or not 0 < args.teleop_hz <= 100:
        parser.error("--teleop-hz must be greater than 0 and at most 100")
    if not sys.stdin.isatty():
        parser.error("An interactive terminal is required")
    with terminal_keys() as read_key:
        try:
            CollectionSession(args).run(read_key)
        except (KeyboardInterrupt, EOFError):
            print("\nCollection session stopped; unfinished plan remains pending.")
        except (OSError, ValueError, RuntimeError) as exc:
            parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
