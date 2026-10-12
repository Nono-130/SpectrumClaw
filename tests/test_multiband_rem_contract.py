"""Contract tests for the 128 x 128 x 5 multi-band spectrum REM."""

from __future__ import annotations

import numpy as np
import pytest


def _valid_sample(spec):
    rss = np.zeros(spec.shape, dtype=np.float32)
    for band_index in range(spec.band_count):
        rss[:, :, band_index] = -80.0 + band_index
    sampled = np.full(spec.shape, np.nan, dtype=np.float32)
    observed = np.zeros(spec.shape, dtype=np.uint8)
    sampled[10, 20, :] = rss[10, 20, :]
    observed[10, 20, :] = 1
    trajectory_xy = np.asarray([spec.row_column_to_enu(10, 20)], dtype=np.float32)
    return {
        "schema_version": spec.schema_version,
        "axis_order": spec.axis_order,
        "rss_dbm": rss,
        "sampled_rss_dbm": sampled,
        "observed_mask": observed,
        "frequencies_hz": np.asarray(spec.frequencies_hz, dtype=np.float64),
        "building_mask": np.zeros((spec.rows, spec.columns), dtype=np.uint8),
        "flyable_mask": np.ones((spec.rows, spec.columns), dtype=np.uint8),
        "valid_rx_mask": np.ones((spec.rows, spec.columns), dtype=np.uint8),
        "bounds_enu_m": np.asarray(spec.bounds_enu_m, dtype=np.float32),
        "cell_size_m": np.asarray(spec.cell_size_m, dtype=np.float32),
        "receiver_altitude_m": np.asarray(spec.receiver_altitude_m, dtype=np.float32),
        "source_xy_m": np.asarray([[12.25, -7.5]], dtype=np.float32),
        "source_power_dbm": np.full((1, spec.band_count), 20.0, dtype=np.float32),
        "source_band_mask": np.ones((1, spec.band_count), dtype=np.uint8),
        "trajectory_xy_m": trajectory_xy,
        "trajectory_rc": np.asarray([[10, 20]], dtype=np.int16),
        "measurements_dbm": rss[10:11, 20, :].astype(np.float32),
        "scene_id": np.asarray("scene-001"),
        "seed": np.asarray(7, dtype=np.int64),
        "split_group_id": np.asarray("source-block-001"),
        "split": np.asarray("train"),
        "sampling_ratio": np.asarray(1.0 / (spec.rows * spec.columns), dtype=np.float32),
        "requested_sampling_ratio": np.asarray(1.0 / (spec.rows * spec.columns), dtype=np.float32),
        "sampling_mode": np.asarray("random_points"),
        "mask_seed": np.asarray(17, dtype=np.int64),
    }


def test_default_contract_is_spatial_hwf_with_five_frequency_bands():
    from backend.skills.spectrum_construction.multiband_rem_contract import DEFAULT_REM_SPEC

    spec = DEFAULT_REM_SPEC

    assert spec.shape == (128, 128, 5)
    assert spec.axis_order == "HWF"
    assert spec.receiver_altitude_m == 21.0
    assert spec.cell_size_m == (0.9375, 0.9375)
    assert len(spec.frequencies_hz) == 5
    assert tuple(sorted(spec.frequencies_hz)) == spec.frequencies_hz
    assert spec.frequencies_hz[0] >= 1.0e9
    assert 2.4e9 in spec.frequencies_hz
    assert "height_levels" not in spec.metadata()


def test_v1_contract_rejects_other_spatial_shapes_or_receiver_altitudes():
    from backend.skills.spectrum_construction.multiband_rem_contract import MultibandRemSpec

    with pytest.raises(ValueError, match="128"):
        MultibandRemSpec(rows=64)
    with pytest.raises(ValueError, match="128"):
        MultibandRemSpec(columns=64)
    with pytest.raises(ValueError, match="21"):
        MultibandRemSpec(receiver_altitude_m=20.0)
    with pytest.raises(ValueError, match="schema_version"):
        MultibandRemSpec(schema_version="spectrumclaw.rem.hwf.v2")


def test_enu_grid_conversion_uses_row_zero_as_north_and_round_trips_cell_centres():
    from backend.skills.spectrum_construction.multiband_rem_contract import DEFAULT_REM_SPEC

    spec = DEFAULT_REM_SPEC

    assert spec.enu_to_row_column(-59.53125, 59.53125) == (0, 0)
    assert spec.enu_to_row_column(59.53125, -59.53125) == (127, 127)
    for row, column in ((0, 0), (31, 79), (127, 127)):
        east, north = spec.row_column_to_enu(row, column)
        assert spec.enu_to_row_column(east, north) == (row, column)

    with pytest.raises(ValueError, match="边界"):
        spec.enu_to_row_column(60.0, 0.0)


def test_sample_validation_enforces_hwf_nan_mask_and_xy_only_labels():
    from backend.skills.spectrum_construction.multiband_rem_contract import (
        DEFAULT_REM_SPEC,
        validate_rem_sample,
    )

    sample = _valid_sample(DEFAULT_REM_SPEC)
    validate_rem_sample(sample, DEFAULT_REM_SPEC)

    wrong_axis = dict(sample, axis_order="ZHW")
    with pytest.raises(ValueError, match="HWF"):
        validate_rem_sample(wrong_axis, DEFAULT_REM_SPEC)

    leaked_height = dict(sample, source_z_m=np.asarray([1.5], dtype=np.float32))
    with pytest.raises(ValueError, match="XY-only"):
        validate_rem_sample(leaked_height, DEFAULT_REM_SPEC)

    leaked_emitter_height = dict(sample, emitter_z_m=np.asarray([1.5], dtype=np.float32))
    with pytest.raises(ValueError, match="XY-only"):
        validate_rem_sample(leaked_emitter_height, DEFAULT_REM_SPEC)

    undeclared_field = dict(sample, future_height_label=np.asarray([1.5], dtype=np.float32))
    with pytest.raises(ValueError, match="未声明字段"):
        validate_rem_sample(undeclared_field, DEFAULT_REM_SPEC)

    inconsistent_mask = dict(sample)
    inconsistent_mask["sampled_rss_dbm"] = sample["sampled_rss_dbm"].copy()
    inconsistent_mask["sampled_rss_dbm"][0, 0, 0] = -55.0
    with pytest.raises(ValueError, match="observed_mask"):
        validate_rem_sample(inconsistent_mask, DEFAULT_REM_SPEC)


def test_sample_validation_cross_checks_trajectory_measurements_and_sparse_tensor():
    from backend.skills.spectrum_construction.multiband_rem_contract import (
        DEFAULT_REM_SPEC,
        validate_rem_sample,
    )

    sample = _valid_sample(DEFAULT_REM_SPEC)

    wrong_xy = dict(sample, trajectory_xy_m=sample["trajectory_xy_m"] + 1.0)
    with pytest.raises(ValueError, match="trajectory_xy_m"):
        validate_rem_sample(wrong_xy, DEFAULT_REM_SPEC)

    wrong_measurement = dict(sample, measurements_dbm=sample["measurements_dbm"] + 1.0)
    with pytest.raises(ValueError, match="measurements_dbm"):
        validate_rem_sample(wrong_measurement, DEFAULT_REM_SPEC)

    extra_observation = dict(sample)
    extra_observation["observed_mask"] = sample["observed_mask"].copy()
    extra_observation["sampled_rss_dbm"] = sample["sampled_rss_dbm"].copy()
    extra_observation["observed_mask"][11, 21, :] = 1
    extra_observation["sampled_rss_dbm"][11, 21, :] = sample["rss_dbm"][11, 21, :]
    with pytest.raises(ValueError, match="trajectory_rc"):
        validate_rem_sample(extra_observation, DEFAULT_REM_SPEC)

    wrong_sparse_value = dict(sample)
    wrong_sparse_value["sampled_rss_dbm"] = sample["sampled_rss_dbm"].copy()
    wrong_sparse_value["sampled_rss_dbm"][10, 20, 0] += 1.0
    with pytest.raises(ValueError, match="sampled_rss_dbm"):
        validate_rem_sample(wrong_sparse_value, DEFAULT_REM_SPEC)


def test_sample_validation_requires_deterministic_group_split():
    from backend.skills.spectrum_construction.multiband_rem_contract import (
        DEFAULT_REM_SPEC,
        validate_rem_sample,
    )

    sample = _valid_sample(DEFAULT_REM_SPEC)

    missing_split = dict(sample)
    missing_split.pop("split")
    with pytest.raises(ValueError, match="split"):
        validate_rem_sample(missing_split, DEFAULT_REM_SPEC)

    with pytest.raises(ValueError, match="split_group_id"):
        validate_rem_sample(dict(sample, split=np.asarray("test")), DEFAULT_REM_SPEC)


def test_sample_validation_rejects_split_directory_mismatch(tmp_path):
    from backend.skills.spectrum_construction.multiband_rem_contract import (
        DEFAULT_REM_SPEC,
        load_rem_sample,
        save_rem_sample,
    )

    sample = _valid_sample(DEFAULT_REM_SPEC)
    wrong_path = tmp_path / "test" / "sample.npz"
    with pytest.raises(ValueError, match="目录"):
        save_rem_sample(wrong_path, sample, DEFAULT_REM_SPEC)

    wrong_path.parent.mkdir(parents=True)
    np.savez_compressed(wrong_path, **sample)
    with pytest.raises(ValueError, match="目录"):
        load_rem_sample(wrong_path, DEFAULT_REM_SPEC)


def test_sample_validation_requires_flyable_valid_receiver_trajectory():
    from backend.skills.spectrum_construction.multiband_rem_contract import (
        DEFAULT_REM_SPEC,
        validate_rem_sample,
    )

    sample = _valid_sample(DEFAULT_REM_SPEC)

    invalid_subset = dict(sample)
    invalid_subset["valid_rx_mask"] = sample["valid_rx_mask"].copy()
    invalid_subset["valid_rx_mask"][0, 0] = 0
    with pytest.raises(ValueError, match="flyable_mask"):
        validate_rem_sample(invalid_subset, DEFAULT_REM_SPEC)

    blocked_trajectory = dict(sample)
    blocked_trajectory["flyable_mask"] = sample["flyable_mask"].copy()
    blocked_trajectory["valid_rx_mask"] = sample["valid_rx_mask"].copy()
    blocked_trajectory["flyable_mask"][10, 20] = 0
    blocked_trajectory["valid_rx_mask"][10, 20] = 0
    with pytest.raises(ValueError, match="轨迹"):
        validate_rem_sample(blocked_trajectory, DEFAULT_REM_SPEC)


def test_npz_round_trip_preserves_versioned_multiband_contract(tmp_path):
    from backend.skills.spectrum_construction.multiband_rem_contract import (
        DEFAULT_REM_SPEC,
        load_rem_sample,
        save_rem_sample,
    )

    sample = _valid_sample(DEFAULT_REM_SPEC)
    path = tmp_path / "sample.npz"

    save_rem_sample(path, sample, DEFAULT_REM_SPEC)
    loaded = load_rem_sample(path, DEFAULT_REM_SPEC)

    assert loaded["axis_order"].item() == "HWF"
    assert loaded["rss_dbm"].shape == (128, 128, 5)
    assert loaded["source_xy_m"].shape == (1, 2)
    np.testing.assert_array_equal(loaded["observed_mask"], sample["observed_mask"])


def test_split_assignment_is_stable_and_group_scoped():
    from backend.skills.spectrum_construction.multiband_rem_contract import assign_split

    first = assign_split("scene-a/source-block-7", seed=123)
    second = assign_split("scene-a/source-block-7", seed=123)

    assert first == second
    assert first in {"train", "validation", "test"}
