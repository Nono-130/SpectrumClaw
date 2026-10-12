"""Training-pipeline tests for versioned multi-band REM NPZ samples."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
import subprocess
import sys


torch = pytest.importorskip("torch")


def _sample(spec):
    rows, columns, bands = spec.shape
    row_grid = np.arange(rows, dtype=np.float32)[:, None, None]
    column_grid = np.arange(columns, dtype=np.float32)[None, :, None]
    band_grid = np.arange(bands, dtype=np.float32)[None, None, :]
    rss = -100.0 + row_grid * 0.1 + column_grid * 0.01 + band_grid
    observed = np.zeros(spec.shape, dtype=np.uint8)
    observed[::16, ::16, :] = 1
    sampled = np.where(observed == 1, rss, np.nan).astype(np.float32)
    trajectory_rc = np.argwhere(observed[:, :, 0] == 1).astype(np.int16)
    trajectory_xy = np.asarray(
        [spec.row_column_to_enu(int(row), int(column)) for row, column in trajectory_rc],
        dtype=np.float32,
    )
    return {
        "schema_version": np.asarray(spec.schema_version),
        "axis_order": np.asarray(spec.axis_order),
        "rss_dbm": rss.astype(np.float32),
        "sampled_rss_dbm": sampled,
        "observed_mask": observed,
        "frequencies_hz": np.asarray(spec.frequencies_hz, dtype=np.float64),
        "building_mask": np.zeros((rows, columns), dtype=np.uint8),
        "flyable_mask": np.ones((rows, columns), dtype=np.uint8),
        "valid_rx_mask": np.ones((rows, columns), dtype=np.uint8),
        "bounds_enu_m": np.asarray(spec.bounds_enu_m, dtype=np.float32),
        "cell_size_m": np.asarray(spec.cell_size_m, dtype=np.float32),
        "receiver_altitude_m": np.asarray(spec.receiver_altitude_m, dtype=np.float32),
        "source_xy_m": np.asarray([[8.5, -11.25]], dtype=np.float32),
        "source_power_dbm": np.full((1, bands), 20.0, dtype=np.float32),
        "source_band_mask": np.ones((1, bands), dtype=np.uint8),
        "trajectory_xy_m": trajectory_xy,
        "trajectory_rc": trajectory_rc,
        "measurements_dbm": rss[trajectory_rc[:, 0], trajectory_rc[:, 1], :].astype(np.float32),
        "scene_id": np.asarray("scene-training-smoke"),
        "seed": np.asarray(13, dtype=np.int64),
        "split_group_id": np.asarray("source-training-smoke"),
        "split": np.asarray("train"),
        "sampling_ratio": np.asarray(len(trajectory_rc) / (rows * columns), dtype=np.float32),
        "requested_sampling_ratio": np.asarray(len(trajectory_rc) / (rows * columns), dtype=np.float32),
        "sampling_mode": np.asarray("random_points"),
        "mask_seed": np.asarray(29, dtype=np.int64),
    }


def test_idw_prior_is_dense_finite_and_preserves_measured_values():
    from backend.skills.spectrum_construction.multiband_rem_contract import DEFAULT_REM_SPEC
    from backend.skills.spectrum_construction.multiband_rem_training import build_idw_prior

    sample = _sample(DEFAULT_REM_SPEC)
    prior = build_idw_prior(sample["sampled_rss_dbm"], sample["observed_mask"])

    assert prior.shape == (128, 128, 5)
    assert np.isfinite(prior).all()
    np.testing.assert_allclose(
        prior[sample["observed_mask"] == 1],
        sample["sampled_rss_dbm"][sample["observed_mask"] == 1],
    )


def test_npz_dataset_returns_model_ready_hwf_tensors_and_xy_target(tmp_path):
    from backend.skills.spectrum_construction.multiband_rem_contract import (
        DEFAULT_REM_SPEC,
        save_rem_sample,
    )
    from backend.skills.spectrum_construction.multiband_rem_training import MultibandRemDataset

    path = save_rem_sample(tmp_path / "sample.npz", _sample(DEFAULT_REM_SPEC), DEFAULT_REM_SPEC)
    dataset = MultibandRemDataset([path], band_dropout_probability=0.0, seed=5)

    item = dataset[0]

    assert item["sparse_rss_dbm"].shape == (128, 128, 5)
    assert item["observed_mask"].shape == (128, 128, 5)
    assert item["physics_prior_dbm"].shape == (128, 128, 5)
    assert item["target_rem_dbm"].shape == (128, 128, 5)
    assert item["building_mask"].shape == (128, 128)
    assert item["frequencies_hz"].shape == (5,)
    assert item["band_available_mask"].shape == (5,)
    assert item["source_xy_m"].shape == (2,)
    assert "source_z_m" not in item
    assert torch.isfinite(item["sparse_rss_dbm"]).all()
    assert torch.isfinite(item["physics_prior_dbm"]).all()


def test_band_dropout_keeps_at_least_one_real_frequency_channel(tmp_path):
    from backend.skills.spectrum_construction.multiband_rem_contract import (
        DEFAULT_REM_SPEC,
        save_rem_sample,
    )
    from backend.skills.spectrum_construction.multiband_rem_training import MultibandRemDataset

    path = save_rem_sample(tmp_path / "sample.npz", _sample(DEFAULT_REM_SPEC), DEFAULT_REM_SPEC)
    dataset = MultibandRemDataset([path], band_dropout_probability=1.0, seed=9)

    item = dataset[0]

    available = item["band_available_mask"] > 0.5
    assert 1 <= int(available.sum()) < 5
    assert torch.all(item["observed_mask"][:, :, ~available] == 0)


def test_band_dropout_changes_by_epoch_but_is_reproducible_for_same_epoch(tmp_path):
    from backend.skills.spectrum_construction.multiband_rem_contract import (
        DEFAULT_REM_SPEC,
        save_rem_sample,
    )
    from backend.skills.spectrum_construction.multiband_rem_training import MultibandRemDataset

    path = save_rem_sample(tmp_path / "sample.npz", _sample(DEFAULT_REM_SPEC), DEFAULT_REM_SPEC)
    dataset = MultibandRemDataset([path], band_dropout_probability=1.0, seed=9)

    masks = []
    for epoch in range(8):
        dataset.set_epoch(epoch)
        masks.append(tuple(dataset[0]["band_available_mask"].tolist()))

    dataset.set_epoch(3)
    repeated = tuple(dataset[0]["band_available_mask"].tolist())
    assert repeated == masks[3]
    assert len(set(masks)) > 1


def test_training_cli_direct_entrypoint_loads_repository_modules():
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/training/train_multiband_rem.py", "--help"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--data" in completed.stdout
