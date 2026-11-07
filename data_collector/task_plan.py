"""Create and resolve a task's collection plan."""

import argparse
import csv
import os
import re
import tempfile
from pathlib import Path


REQUIRED_COLUMNS = {"task_id", "planned_episode_id", "planned_order", "split"}
PLAN_SIZE = 200
LEGACY_PLAN_SIZE = 70
LEGACY_TRAIN_SIZE = 50
TRAIN_SIZE = 170
VAL_SIZE = 30


def read_manifest(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(f"Invalid manifest columns: {path}")
        return reader.fieldnames, list(reader)


def write_manifest_atomic(path, fields, rows):
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


def custom_plan_rows(fields, train_size, val_size):
    total = train_size + val_size
    rows = []
    train_index = 0
    val_index = train_size
    for order in range(1, total + 1):
        is_val = order * val_size // total > (order - 1) * val_size // total
        row = dict.fromkeys(fields, "")
        row.update(task_id="TASK_ID_TODO", planned_episode_id=f"PLAN-{order:03d}",
                   planned_order=str(order), acceptance_status="pending",
                   split="val" if is_val else "train")
        if is_val:
            row["planned_export_episode_index"] = str(val_index)
            val_index += 1
        else:
            row["planned_export_episode_index"] = str(train_index)
            train_index += 1
        rows.append(row)
    return rows


def ensure_manifest(template, manifest, task_id,
                    train_size=TRAIN_SIZE, val_size=VAL_SIZE):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", task_id):
        raise ValueError("task-id must be one safe path component")
    if train_size < 1 or val_size < 1:
        raise ValueError("Training and validation episode counts must be positive")

    template_fields, template_rows = read_manifest(template)
    if (len(template_rows) != PLAN_SIZE or
            sum(row["split"] == "train" for row in template_rows) != TRAIN_SIZE or
            sum(row["split"] == "val" for row in template_rows) != VAL_SIZE or
            any(row["task_id"] != "TASK_ID_TODO" or
                row["planned_episode_id"] != f"PLAN-{index:03d}" or
                row["planned_order"] != str(index)
                for index, row in enumerate(template_rows, 1)) or
            {row["planned_export_episode_index"] for row in template_rows} !=
            {str(index) for index in range(PLAN_SIZE)}):
        raise ValueError(f"Expected the unfilled {PLAN_SIZE}-row plan template")

    default_plan = (train_size, val_size) == (TRAIN_SIZE, VAL_SIZE)
    if not default_plan:
        template_rows = custom_plan_rows(template_fields, train_size, val_size)
    plan_size = train_size + val_size

    if not manifest.exists():
        for row in template_rows:
            row["task_id"] = task_id
            row["accepted_episode_id"] = ""
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=template_fields)
            writer.writeheader()
            writer.writerows(template_rows)

    fields, rows = read_manifest(manifest)
    if len(rows) != plan_size and not (default_plan and len(rows) == LEGACY_PLAN_SIZE):
        raise ValueError(f"Manifest must contain {plan_size} unique plan rows")
    fixed_columns = ("planned_episode_id", "planned_order", "split",
                     "planned_export_episode_index")
    for row, template_row in zip(rows, template_rows):
        expected = dict(template_row)
        if default_plan and len(rows) == LEGACY_PLAN_SIZE and expected["split"] == "val":
            expected["planned_export_episode_index"] = str(
                int(expected["planned_export_episode_index"]) -
                (TRAIN_SIZE - LEGACY_TRAIN_SIZE))
        if row["task_id"] != task_id or any(
                row.get(column) != expected.get(column) for column in fixed_columns):
            raise ValueError("Manifest task or fixed plan columns differ from template")
    if default_plan and len(rows) == LEGACY_PLAN_SIZE:
        for row, template_row in zip(rows, template_rows):
            if row["split"] == "val":
                old_index = row["planned_export_episode_index"]
                row["planned_export_episode_index"] = template_row["planned_export_episode_index"]
                if row.get("accepted_episode_id") == old_index:
                    row["accepted_episode_id"] = (
                        template_row["planned_export_episode_index"]
                        if row.get("acceptance_status") == "accepted" else "")
            if (row.get("acceptance_status") != "accepted" and
                    row.get("accepted_episode_id") == row["planned_export_episode_index"]):
                row["accepted_episode_id"] = ""
        for template_row in template_rows[LEGACY_PLAN_SIZE:]:
            row = dict(template_row)
            row["task_id"] = task_id
            row["accepted_episode_id"] = ""
            rows.append(row)
        write_manifest_atomic(manifest, fields, rows)
    return fields, rows


def resolve_plan(template, manifest, task_id, plan_number,
                 train_size=TRAIN_SIZE, val_size=VAL_SIZE):
    plan_size = train_size + val_size
    if not 1 <= plan_number <= plan_size:
        raise ValueError(f"plan number must be between 1 and {plan_size}")
    _, rows = ensure_manifest(template, manifest, task_id, train_size, val_size)
    planned_id = f"PLAN-{plan_number:03d}"
    row = rows[plan_number - 1]
    return planned_id, row["split"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--train-episodes", type=int, default=TRAIN_SIZE)
    parser.add_argument("--val-episodes", type=int, default=VAL_SIZE)
    parser.add_argument("plan_number", type=int)
    args = parser.parse_args()
    try:
        planned_id, split = resolve_plan(
            args.template, args.manifest, args.task_id, args.plan_number,
            args.train_episodes, args.val_episodes)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(planned_id, split)


if __name__ == "__main__":
    main()
