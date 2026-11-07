"""Read-only training/deployment parity checks; export one observation for WS testing.

Does not load model weights, run inference, or regenerate dataset statistics.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from policy_server import DEFAULT_PROMPT, ROOT, FR3Normalizer, FR3PolicyService, inspect_checkpoint


def verify(dataset_dir: Path, ckpt_dir: Path, output_dir: Path) -> dict:
    from omegaconf import OmegaConf

    from benchmarks.utils.client import build_payload, encode_numpy_b64
    from openwam.dataloader.fr3 import FR3Dataset, dataset_signature
    from openwam.dataloader.utils.video_io import decode_video_frames
    from openwam.deploy.model_loader import _build_normalizer
    from openwam.deploy.obs_preprocess import ObsPreprocessor

    cfg = OmegaConf.load(ckpt_dir / "config.yaml")
    metadata = inspect_checkpoint(ckpt_dir, None, 8)
    cfg.dataloader.dataset_dir = str(dataset_dir)
    saved = np.load(ckpt_dir / "normalization_stats.npy", allow_pickle=True).item()
    dataset_stats = np.load(dataset_dir / "meta/normalization_stats.npy", allow_pickle=True).item()
    signature = dataset_signature(dataset_dir)
    if saved.get("dataset_sha256") != signature or dataset_stats.get("dataset_sha256") != signature:
        raise ValueError("Dataset signature differs from the checkpoint/training statistics")
    for key in ("eef", "eef_state"):
        if saved[key].keys() != dataset_stats[key].keys() or any(
            not np.array_equal(saved[key][field], dataset_stats[key][field]) for field in saved[key]
        ):
            raise ValueError(f"Checkpoint {key} statistics differ from the supplied dataset")

    normalizer = FR3Normalizer(_build_normalizer(cfg, str(ckpt_dir)))
    service = FR3PolicyService(None, metadata, DEFAULT_PROMPT)
    preprocessor = ObsPreprocessor.from_cfg(cfg)
    info = json.loads((dataset_dir / "meta/info.json").read_text())
    report = {
        "dataset_dir": str(dataset_dir),
        "checkpoint_dir": str(ckpt_dir),
        "dataset_sha256": signature,
        "total_episodes": info["total_episodes"],
        "total_frames": info["total_frames"],
        "action_fps": info["fps"],
        "image_max_abs_error": 0,
        "state_max_abs_error": 0.0,
        "action_inverse_max_abs_error": 0.0,
        "samples": [],
        "scope": "CPU preprocessing/normalization only; no model forward or robot execution",
    }
    first_observation = None
    for split in ("train", "val"):
        dataset = FR3Dataset.from_config(cfg.dataloader, split=split)
        for index in sorted({0, min(1, len(dataset) - 1), len(dataset) - 1}):
            sample = dataset[index]
            ep_local = int(np.searchsorted(dataset._cum_n_starts, index, side="right") - 1)
            row = dataset._eps_df.iloc[ep_local]
            offset = (index - int(dataset._cum_n_starts[ep_local])) * dataset._window_stride
            table = dataset._load_data_table(int(row["data/chunk_index"]), int(row["data/file_index"]))
            start = int(dataset._ep_data_row_offset[ep_local]) + offset
            raw_state = table.slice(start, 1).to_pandas()["observation.state"].iloc[0].tolist()
            images = {}
            for camera in metadata["required_cameras"]:
                key = f"observation.images.{camera}"
                path = dataset_dir / info["video_path"].format(
                    video_key=key,
                    chunk_index=int(row[f"videos/{key}/chunk_index"]),
                    file_index=int(row[f"videos/{key}/file_index"]),
                )
                frame_index = int(dataset._ep_video_frame_offsets[key][ep_local]) + offset
                height, width, _ = info["features"][key]["shape"]
                # Native-size decode keeps the online server responsible for resizing.
                images[camera] = decode_video_frames(str(path), [frame_index], height, width)[0]
            payload = build_payload(
                head=encode_numpy_b64(np.asarray(images["head_camera"])),
                left_wrist=encode_numpy_b64(np.asarray(images["left_wrist_camera"])),
                state=raw_state,
                prompt=sample["prompt"],
            )
            online = preprocessor.preprocess(service.prepare_observation(payload))
            image_error = int(
                np.abs(
                    np.asarray(online["image"], dtype=np.int16)
                    - np.asarray(sample["first_frame_image"][0], dtype=np.int16)
                ).max()
            )
            state = normalizer.normalize(np.asarray(raw_state, dtype=np.float32)[None])
            state_error = float(np.abs(state - sample["proprio"].numpy()).max())
            normalized_actions = sample["action"].numpy()
            actual = normalizer.unnormalize(normalized_actions)
            expected = dataset.denormalize_action(normalized_actions)
            action_error = float(np.abs(actual - expected).max())
            if image_error != 0 or state_error != 0 or action_error > 3e-7:
                raise ValueError(f"{split}/{index}: image={image_error}, state={state_error}, action={action_error}")
            report["image_max_abs_error"] = max(report["image_max_abs_error"], image_error)
            report["state_max_abs_error"] = max(report["state_max_abs_error"], state_error)
            report["action_inverse_max_abs_error"] = max(report["action_inverse_max_abs_error"], action_error)
            report["samples"].append({"split": split, "episode": int(row.episode_index), "offset": int(offset)})
            if first_observation is None:
                if sample["prompt"] != DEFAULT_PROMPT:
                    raise ValueError("Default prompt differs from the recorded stack-bowl task")
                first_observation = images, raw_state, sample["prompt"]

    output_dir.mkdir(parents=True, exist_ok=True)
    images, state, prompt = first_observation
    for camera, image in images.items():
        image.save(output_dir / f"{camera}.png")
    (output_dir / "state.json").write_text(json.dumps(state, indent=2) + "\n")
    (output_dir / "prompt.txt").write_text(prompt + "\n")
    report["status"] = "passed"
    (output_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "assets/data/fr3_data/stack_bowl")
    parser.add_argument("--ckpt-dir", type=Path, default=ROOT / "assets/fr3_stack_bowl_ckpt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/fr3_real_inference")
    args = parser.parse_args()
    report = verify(*(p.expanduser().resolve() for p in (args.dataset_dir, args.ckpt_dir, args.output_dir)))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
