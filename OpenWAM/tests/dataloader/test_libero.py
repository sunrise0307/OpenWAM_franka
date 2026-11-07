from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from openwam.dataloader.libero import LiberoDataset
from openwam.dataloader.registry import DATASET_REGISTRY


def test_canonical_config_and_registry_use_native_action_reader() -> None:
    config = yaml.safe_load(Path("configs/dataloader/libero.yaml").read_text(encoding="utf-8"))
    assert config["type"] == "libero"
    assert config["action_mode"] == "eef"
    assert DATASET_REGISTRY["libero"] is LiberoDataset


def test_reader_hard_preserves_rot6d_under_custom_stats() -> None:
    reader = object.__new__(LiberoDataset)
    reader._normalize_mode = "min-max"
    # Deliberately non-identity rotation stats: the reader must still leave
    # rot6d untouched rather than relying only on the generated stats artifact.
    reader._normalization_stats = {
        "min": np.zeros(10, dtype=np.float32),
        "max": np.ones(10, dtype=np.float32),
    }
    raw = np.array(
        [[0.25, 0.5, 0.75, 0.998, 0.02, -0.04, 0.01, 0.999, 0.02, -0.5]],
        dtype=np.float32,
    )
    normalized = reader._normalize_array(raw)
    np.testing.assert_array_equal(normalized[:, 3:9], raw[:, 3:9])
    expected_non_rot = np.clip(raw[:, [0, 1, 2, 9]] * 2.0 - 1.0, -1.0, 1.0)
    np.testing.assert_allclose(normalized[:, [0, 1, 2, 9]], expected_non_rot)


def test_reader_rejects_incomplete_compatibility_stats(tmp_path) -> None:
    reader = object.__new__(LiberoDataset)
    reader._normalize_mode = "min-max"
    reader._source_stats_path = str(tmp_path / "normalization_stats.npy")
    reader._dataset_dir = tmp_path
    reader._raw_action_dim = 10
    np.save(reader._source_stats_path, {"eef": {}}, allow_pickle=True)
    with pytest.raises(KeyError, match="eef"):
        reader._load_stats({})


def test_multiview_target_camera_does_not_disable_wrist_stream() -> None:
    reader = object.__new__(LiberoDataset)
    reader._multiview = True
    reader._target_camera = "observation.images.image"
    reader._camera_layout_param = [
        "observation.images.image",
        "observation.images.image2",
        None,
    ]
    reader._head_priority = LiberoDataset.HEAD_CAMERA_PRIORITY
    reader._wrist_priority = LiberoDataset.WRIST_CAMERA_PRIORITY

    features = {
        "observation.images.image": {"dtype": "video"},
        "observation.images.image2": {"dtype": "video"},
    }
    assert reader._resolve_cameras({"features": features}) == (
        "observation.images.image",
        "observation.images.image2",
        None,
    )


def test_libero_camera_resolution_falls_back_to_info_features() -> None:
    reader = object.__new__(LiberoDataset)
    reader._multiview = True
    reader._target_camera = "observation.images.image"
    reader._camera_layout_param = None
    reader._head_priority = LiberoDataset.HEAD_CAMERA_PRIORITY
    reader._wrist_priority = LiberoDataset.WRIST_CAMERA_PRIORITY

    features = {
        "observation.images.agentview_image": {"dtype": "video"},
        "observation.images.wrist_image": {"dtype": "video"},
    }
    assert reader._resolve_cameras({"features": features}) == (
        "observation.images.agentview_image",
        "observation.images.wrist_image",
        None,
    )


def test_single_view_target_camera_still_wins() -> None:
    reader = object.__new__(LiberoDataset)
    reader._multiview = False
    reader._target_camera = "observation.images.image2"
    reader._camera_layout_param = None
    reader._head_priority = LiberoDataset.HEAD_CAMERA_PRIORITY
    reader._wrist_priority = LiberoDataset.WRIST_CAMERA_PRIORITY

    assert reader._resolve_cameras({"features": {}}) == (
        "observation.images.image2",
        None,
        None,
    )


def test_explicit_layout_falls_back_to_legacy_wrist_feature() -> None:
    reader = object.__new__(LiberoDataset)
    reader._multiview = True
    reader._target_camera = "observation.images.image"
    reader._camera_layout_param = [
        "observation.images.image",
        "observation.images.image2",
        None,
    ]
    reader._head_priority = LiberoDataset.HEAD_CAMERA_PRIORITY
    reader._wrist_priority = LiberoDataset.WRIST_CAMERA_PRIORITY

    features = {
        "observation.images.image": {"dtype": "video"},
        "observation.images.wrist_image": {"dtype": "video"},
    }
    assert reader._resolve_cameras({"features": features}) == (
        "observation.images.image",
        "observation.images.wrist_image",
        None,
    )
    assert reader._camera_layout_param == [
        "observation.images.image",
        "observation.images.wrist_image",
        None,
    ]


def test_single_view_explicit_layout_is_honored_without_target() -> None:
    reader = object.__new__(LiberoDataset)
    reader._multiview = False
    reader._target_camera = None
    reader._camera_layout_param = ["observation.images.agentview_image", None, None]
    reader._head_priority = LiberoDataset.HEAD_CAMERA_PRIORITY
    reader._wrist_priority = LiberoDataset.WRIST_CAMERA_PRIORITY

    features = {
        "observation.images.image": {"dtype": "video"},
        "observation.images.agentview_image": {"dtype": "video"},
    }
    assert reader._resolve_cameras({"features": features}) == (
        "observation.images.agentview_image",
        None,
        None,
    )
