"""Tests for deterministic five-band offline REM sample generation."""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace


class _FakeDenseMapProvider:
    def __init__(self):
        self.calls = []

    def generate_band_map(self, *, spec, frequency_hz, source_xy_m, source_power_dbm, seed):
        self.calls.append((frequency_hz, tuple(source_xy_m), source_power_dbm, seed))
        band_index = spec.frequencies_hz.index(frequency_hz)
        rows = np.arange(spec.rows, dtype=np.float32)[:, None]
        columns = np.arange(spec.columns, dtype=np.float32)[None, :]
        return -100.0 + band_index + rows * 0.01 + columns * 0.001


def test_generator_preserves_each_frequency_as_a_separate_hwf_channel():
    from backend.skills.spectrum_construction.multiband_rem_contract import DEFAULT_REM_SPEC
    from simulation.rf_engine.sionna_rem_dataset import generate_rem_sample

    provider = _FakeDenseMapProvider()
    sample = generate_rem_sample(
        provider=provider,
        spec=DEFAULT_REM_SPEC,
        seed=19,
        sampling_ratio=0.01,
        trajectory_mode="random_points",
    )

    assert sample["rss_dbm"].shape == (128, 128, 5)
    assert sample["sampled_rss_dbm"].shape == (128, 128, 5)
    assert sample["observed_mask"].shape == (128, 128, 5)
    assert [call[0] for call in provider.calls] == list(DEFAULT_REM_SPEC.frequencies_hz)
    assert np.allclose(sample["rss_dbm"][0, 0, :], [-100, -99, -98, -97, -96])
    assert sample["receiver_altitude_m"].item() == 21.0
    assert sample["source_xy_m"].shape == (1, 2)
    assert "source_z_m" not in sample


def test_same_seed_reproduces_source_trajectory_and_sparse_measurements():
    from backend.skills.spectrum_construction.multiband_rem_contract import DEFAULT_REM_SPEC
    from simulation.rf_engine.sionna_rem_dataset import generate_rem_sample

    first = generate_rem_sample(
        provider=_FakeDenseMapProvider(),
        spec=DEFAULT_REM_SPEC,
        seed=41,
        sampling_ratio=0.015,
        trajectory_mode="random_walk",
    )
    second = generate_rem_sample(
        provider=_FakeDenseMapProvider(),
        spec=DEFAULT_REM_SPEC,
        seed=41,
        sampling_ratio=0.015,
        trajectory_mode="random_walk",
    )

    np.testing.assert_array_equal(first["source_xy_m"], second["source_xy_m"])
    np.testing.assert_array_equal(first["trajectory_rc"], second["trajectory_rc"])
    np.testing.assert_array_equal(first["observed_mask"], second["observed_mask"])
    np.testing.assert_allclose(first["sampled_rss_dbm"], second["sampled_rss_dbm"], equal_nan=True)


def test_random_walk_reaches_requested_unique_coverage_and_records_actual_ratio():
    from backend.skills.spectrum_construction.multiband_rem_contract import DEFAULT_REM_SPEC
    from simulation.rf_engine.sionna_rem_dataset import generate_rem_sample

    requested_ratio = 0.05
    sample = generate_rem_sample(
        provider=_FakeDenseMapProvider(),
        spec=DEFAULT_REM_SPEC,
        seed=41,
        sampling_ratio=requested_ratio,
        trajectory_mode="random_walk",
    )

    unique_count = len(np.unique(sample["trajectory_rc"], axis=0))
    expected_count = round(DEFAULT_REM_SPEC.rows * DEFAULT_REM_SPEC.columns * requested_ratio)
    assert unique_count == expected_count
    assert len(sample["trajectory_rc"]) >= unique_count
    assert float(sample["sampling_ratio"]) == pytest.approx(
        unique_count / (DEFAULT_REM_SPEC.rows * DEFAULT_REM_SPEC.columns)
    )
    assert float(sample["requested_sampling_ratio"]) == pytest.approx(requested_ratio)


def test_all_five_bands_share_one_spatial_sampling_trajectory():
    from backend.skills.spectrum_construction.multiband_rem_contract import DEFAULT_REM_SPEC
    from simulation.rf_engine.sionna_rem_dataset import generate_rem_sample

    sample = generate_rem_sample(
        provider=_FakeDenseMapProvider(),
        spec=DEFAULT_REM_SPEC,
        seed=5,
        sampling_ratio=0.01,
        trajectory_mode="lawnmower",
    )

    first_band_mask = sample["observed_mask"][:, :, 0]
    for band_index in range(1, DEFAULT_REM_SPEC.band_count):
        np.testing.assert_array_equal(sample["observed_mask"][:, :, band_index], first_band_mask)
    assert int(first_band_mask.sum()) == len(np.unique(sample["trajectory_rc"], axis=0))


def test_dataset_cli_runs_directly_from_the_repository_root(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "simulation/rf_engine/sionna_rem_dataset.py",
            "--config",
            "config/uav_rem_dataset.yaml",
            "--output",
            str(tmp_path / "dataset"),
            "--provider",
            "analytic",
            "--scenes",
            "1",
            "--masks-per-scene",
            "1",
            "--seed",
            "31",
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "dataset" / "manifest.json").is_file()


def test_default_pilot_config_covers_all_splits_needed_for_training():
    from backend.skills.spectrum_construction.multiband_rem_contract import assign_split

    repository_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((repository_root / "config/uav_rem_dataset.yaml").read_text(encoding="utf-8"))
    base_seed = int(config["dataset"]["seed"])
    scene_count = int(config["dataset"]["scenes"])
    masks_per_scene = int(config["dataset"]["masks_per_scene"])
    splits = {
        assign_split(f"urban-block/source-seed-{seed:08d}", seed=0)
        for seed in range(base_seed, base_seed + scene_count)
    }

    assert scene_count >= 50
    assert masks_per_scene >= 8
    assert splits == {"train", "validation", "test"}


def test_sionna_provider_matches_online_propagation_flags(monkeypatch, tmp_path):
    from backend.skills.spectrum_construction.multiband_rem_contract import DEFAULT_REM_SPEC
    from simulation.rf_engine.sionna_rem_dataset import SionnaRadioMapProvider

    captured = {}

    class Scene:
        frequency = None

        def add(self, _item):
            pass

    class Solver:
        def __call__(self, _scene, **kwargs):
            captured.update(kwargs)
            rss = np.full((1, 128, 128), 1e-9, dtype=np.float32)
            return SimpleNamespace(rss=SimpleNamespace(numpy=lambda: rss))

    mitsuba = ModuleType("mitsuba")
    mitsuba.set_variant = lambda _variant: None
    mitsuba.Point3f = lambda *values: values
    mitsuba.Point2f = lambda *values: values
    sionna = ModuleType("sionna")
    sionna_rt = ModuleType("sionna.rt")
    sionna_rt.PlanarArray = lambda **_kwargs: object()
    sionna_rt.RadioMapSolver = Solver
    sionna_rt.Transmitter = lambda **_kwargs: object()
    sionna_rt.load_scene = lambda _path: Scene()
    monkeypatch.setitem(sys.modules, "mitsuba", mitsuba)
    monkeypatch.setitem(sys.modules, "sionna", sionna)
    monkeypatch.setitem(sys.modules, "sionna.rt", sionna_rt)

    provider = SionnaRadioMapProvider(asset_dir=tmp_path)
    result = provider.generate_band_map(
        spec=DEFAULT_REM_SPEC,
        frequency_hz=2.4e9,
        source_xy_m=(0.0, 0.0),
        source_power_dbm=20.0,
        seed=1,
    )

    assert result.shape == (128, 128)
    assert captured["samples_per_tx"] == 10_000_000
    assert captured["max_depth"] == 2
    assert captured["los"] is True
    assert captured["specular_reflection"] is True
    assert captured["diffuse_reflection"] is False
    assert captured["refraction"] is True
    assert captured["diffraction"] is False
