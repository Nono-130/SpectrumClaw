#!/usr/bin/env python3
"""Generate fixed-altitude 128x128x5 spectrum REM training samples.

The online UAV acquisition grid remains unchanged.  This offline generator
uses a dense radio-map provider and stores spatial tensors in HWF order, where
F is frequency rather than height.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.skills.spectrum_construction.multiband_rem_contract import (
    AXIS_ORDER,
    DEFAULT_REM_SPEC,
    MultibandRemSpec,
    SCHEMA_VERSION,
    assign_split,
    save_rem_sample,
    validate_rem_sample,
)
from simulation.rf_engine.sionna_measurement_sidecar import (
    BUILDINGS,
    write_urban_block_scene_assets,
)


class DenseBandMapProvider(Protocol):
    def generate_band_map(
        self,
        *,
        spec: MultibandRemSpec,
        frequency_hz: float,
        source_xy_m: tuple[float, float],
        source_power_dbm: float,
        seed: int,
    ) -> np.ndarray: ...


def _cell_centres(spec: MultibandRemSpec) -> tuple[np.ndarray, np.ndarray]:
    west, _east, _south, north = spec.bounds_enu_m
    cell_east, cell_north = spec.cell_size_m
    east = west + (np.arange(spec.columns, dtype=np.float64) + 0.5) * cell_east
    northing = north - (np.arange(spec.rows, dtype=np.float64) + 0.5) * cell_north
    return np.meshgrid(east, northing)


def _footprint_masks(spec: MultibandRemSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    east_grid, north_grid = _cell_centres(spec)
    building = np.zeros((spec.rows, spec.columns), dtype=np.uint8)
    blocks_at_plane = np.zeros_like(building)
    for _name, center, size in BUILDINGS:
        east, north, centre_height = center
        width, depth, height = size
        footprint = (
            (np.abs(east_grid - east) <= width / 2.0)
            & (np.abs(north_grid - north) <= depth / 2.0)
        )
        building[footprint] = 1
        if centre_height + height / 2.0 >= spec.receiver_altitude_m:
            blocks_at_plane[footprint] = 1

    # The current safe UAV mission envelope is approximately +/-35 m in XY.
    flyable = (
        (np.abs(east_grid) <= 35.0)
        & (np.abs(north_grid) <= 35.0)
        & (blocks_at_plane == 0)
    ).astype(np.uint8)
    valid_rx = (blocks_at_plane == 0).astype(np.uint8)
    return building, flyable, valid_rx


def _point_in_building(east_m: float, north_m: float) -> bool:
    for _name, center, size in BUILDINGS:
        east, north, _height = center
        width, depth, _building_height = size
        if abs(east_m - east) <= width / 2.0 and abs(north_m - north) <= depth / 2.0:
            return True
    return False


def _choose_source_xy(spec: MultibandRemSpec, rng: np.random.Generator) -> tuple[float, float]:
    west, east, south, north = spec.bounds_enu_m
    margin_m = 4.0
    for _attempt in range(2_000):
        candidate = (
            float(rng.uniform(west + margin_m, east - margin_m)),
            float(rng.uniform(south + margin_m, north - margin_m)),
        )
        if not _point_in_building(*candidate):
            return candidate
    raise RuntimeError("无法在 urban_block 场景中采样有效的单辐射源 XY")


def _trajectory_indices(
    *,
    flyable_mask: np.ndarray,
    rng: np.random.Generator,
    sampling_ratio: float,
    mode: str,
) -> np.ndarray:
    if not 0 < sampling_ratio <= 1:
        raise ValueError("sampling_ratio 必须位于 (0, 1]")
    free = np.argwhere(flyable_mask == 1)
    if len(free) == 0:
        raise ValueError("固定高度平面没有可飞行的采样网格")
    requested = max(3, int(round(flyable_mask.size * sampling_ratio)))
    count = min(requested, len(free))

    if mode == "random_points":
        selected = rng.choice(len(free), size=count, replace=False)
        return free[selected].astype(np.int16)

    if mode == "lawnmower":
        ordered: list[np.ndarray] = []
        for row in np.unique(free[:, 0]):
            row_cells = free[free[:, 0] == row]
            row_cells = row_cells[np.argsort(row_cells[:, 1])]
            if len(ordered) % 2:
                row_cells = row_cells[::-1]
            ordered.extend(row_cells)
        path = np.asarray(ordered, dtype=np.int16)
        selected = np.linspace(0, len(path) - 1, num=count, dtype=int)
        return path[selected]

    if mode != "random_walk":
        raise ValueError(f"未知的采样轨迹模式: {mode}")

    free_set = {tuple(int(value) for value in item) for item in free}
    current = tuple(int(value) for value in free[int(rng.integers(0, len(free)))])
    visited = {current}
    stack = [current]
    path = [current]
    neighbor_steps = tuple(
        (row_step, column_step)
        for row_step in (-1, 0, 1)
        for column_step in (-1, 0, 1)
        if row_step or column_step
    )
    while len(visited) < count:
        current = stack[-1]
        candidates = [
            (current[0] + row_step, current[1] + column_step)
            for row_step, column_step in neighbor_steps
            if (current[0] + row_step, current[1] + column_step) in free_set
            and (current[0] + row_step, current[1] + column_step) not in visited
        ]
        if candidates:
            current = candidates[int(rng.integers(0, len(candidates)))]
            visited.add(current)
            stack.append(current)
            path.append(current)
            continue

        stack.pop()
        if stack:
            path.append(stack[-1])
            continue

        # Supports disconnected flyable regions while keeping each component's
        # recorded motion contiguous.  The current urban-block mask is connected.
        remaining = sorted(free_set - visited)
        current = remaining[int(rng.integers(0, len(remaining)))]
        visited.add(current)
        stack.append(current)
        path.append(current)
    return np.asarray(path, dtype=np.int16)


def generate_dense_rem(
    *,
    provider: DenseBandMapProvider,
    spec: MultibandRemSpec = DEFAULT_REM_SPEC,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    source_xy = _choose_source_xy(spec, rng)
    source_power = rng.uniform(14.0, 22.0, size=spec.band_count).astype(np.float32)
    band_maps = []
    for band_index, frequency_hz in enumerate(spec.frequencies_hz):
        band_map = np.asarray(provider.generate_band_map(
            spec=spec,
            frequency_hz=frequency_hz,
            source_xy_m=source_xy,
            source_power_dbm=float(source_power[band_index]),
            seed=int(seed) * 101 + band_index,
        ), dtype=np.float32)
        if band_map.shape != (spec.rows, spec.columns):
            raise ValueError(
                f"稠密频段图形状应为 {(spec.rows, spec.columns)}，实际为 {band_map.shape}"
            )
        if not np.all(np.isfinite(band_map)):
            raise ValueError("稠密频段图包含非有限 RSS")
        band_maps.append(band_map)
    building, flyable, valid_rx = _footprint_masks(spec)
    split_group_id = f"urban-block/source-seed-{int(seed):08d}"
    return {
        "rss_dbm": np.stack(band_maps, axis=-1).astype(np.float32),
        "source_xy_m": np.asarray([source_xy], dtype=np.float32),
        "source_power_dbm": source_power.reshape(1, spec.band_count),
        "source_band_mask": np.ones((1, spec.band_count), dtype=np.uint8),
        "building_mask": building,
        "flyable_mask": flyable,
        "valid_rx_mask": valid_rx,
        "scene_id": np.asarray(f"urban-block-{int(seed):08d}"),
        "seed": np.asarray(int(seed), dtype=np.int64),
        "split_group_id": np.asarray(split_group_id),
        "split": np.asarray(assign_split(split_group_id, seed=0)),
    }


def sample_dense_rem(
    dense: dict[str, Any],
    *,
    spec: MultibandRemSpec = DEFAULT_REM_SPEC,
    seed: int,
    sampling_ratio: float,
    trajectory_mode: str,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    trajectory_rc = _trajectory_indices(
        flyable_mask=np.asarray(dense["flyable_mask"]),
        rng=rng,
        sampling_ratio=sampling_ratio,
        mode=trajectory_mode,
    )
    trajectory_xy = np.asarray(
        [spec.row_column_to_enu(int(row), int(column)) for row, column in trajectory_rc],
        dtype=np.float32,
    )
    rss = np.asarray(dense["rss_dbm"], dtype=np.float32)
    measurements = rss[trajectory_rc[:, 0], trajectory_rc[:, 1], :].astype(np.float32)
    sampled = np.full(spec.shape, np.nan, dtype=np.float32)
    observed = np.zeros(spec.shape, dtype=np.uint8)
    sampled[trajectory_rc[:, 0], trajectory_rc[:, 1], :] = measurements
    observed[trajectory_rc[:, 0], trajectory_rc[:, 1], :] = 1
    actual_sampling_ratio = float(np.count_nonzero(observed[:, :, 0])) / float(spec.rows * spec.columns)
    sample = {
        **dense,
        "schema_version": np.asarray(spec.schema_version),
        "axis_order": np.asarray(AXIS_ORDER),
        "sampled_rss_dbm": sampled,
        "observed_mask": observed,
        "frequencies_hz": np.asarray(spec.frequencies_hz, dtype=np.float64),
        "bounds_enu_m": np.asarray(spec.bounds_enu_m, dtype=np.float32),
        "cell_size_m": np.asarray(spec.cell_size_m, dtype=np.float32),
        "receiver_altitude_m": np.asarray(spec.receiver_altitude_m, dtype=np.float32),
        "trajectory_xy_m": trajectory_xy,
        "trajectory_rc": trajectory_rc,
        "measurements_dbm": measurements,
        "sampling_ratio": np.asarray(actual_sampling_ratio, dtype=np.float32),
        "requested_sampling_ratio": np.asarray(float(sampling_ratio), dtype=np.float32),
        "sampling_mode": np.asarray(trajectory_mode),
        "mask_seed": np.asarray(int(seed), dtype=np.int64),
    }
    validate_rem_sample(sample, spec)
    return sample


def generate_rem_sample(
    *,
    provider: DenseBandMapProvider,
    spec: MultibandRemSpec = DEFAULT_REM_SPEC,
    seed: int,
    sampling_ratio: float,
    trajectory_mode: str,
) -> dict[str, Any]:
    dense = generate_dense_rem(provider=provider, spec=spec, seed=seed)
    return sample_dense_rem(
        dense,
        spec=spec,
        seed=seed,
        sampling_ratio=sampling_ratio,
        trajectory_mode=trajectory_mode,
    )


class AnalyticDenseMapProvider:
    """Fast deterministic smoke provider; not a substitute for Sionna data."""

    def generate_band_map(
        self,
        *,
        spec: MultibandRemSpec,
        frequency_hz: float,
        source_xy_m: tuple[float, float],
        source_power_dbm: float,
        seed: int,
    ) -> np.ndarray:
        east, north = _cell_centres(spec)
        distance = np.maximum(1.0, np.hypot(east - source_xy_m[0], north - source_xy_m[1]))
        free_space_at_one_m = 20.0 * math.log10(4.0 * math.pi * frequency_hz / 299_792_458.0)
        base = source_power_dbm - free_space_at_one_m - 25.0 * np.log10(distance)
        rng = np.random.default_rng(seed)
        shadow = rng.normal(0.0, 1.5, size=(spec.rows, spec.columns))
        return np.clip(base + shadow, -140.0, 20.0).astype(np.float32)


class SionnaRadioMapProvider:
    """Sionna RT RadioMapSolver provider loaded only inside the RT environment."""

    def __init__(
        self,
        *,
        asset_dir: str | Path,
        source_altitude_m: float = 1.5,
        samples_per_tx: int = 10_000_000,
        max_depth: int = 2,
    ) -> None:
        self.asset_dir = Path(asset_dir)
        self.source_altitude_m = float(source_altitude_m)
        self.samples_per_tx = int(samples_per_tx)
        self.max_depth = int(max_depth)

    def generate_band_map(
        self,
        *,
        spec: MultibandRemSpec,
        frequency_hz: float,
        source_xy_m: tuple[float, float],
        source_power_dbm: float,
        seed: int,
    ) -> np.ndarray:
        import mitsuba as mi

        mi.set_variant("cuda_ad_mono_polarized")
        from sionna.rt import PlanarArray, RadioMapSolver, Transmitter, load_scene

        scene = load_scene(str(write_urban_block_scene_assets(self.asset_dir)))
        scene.frequency = float(frequency_hz)
        scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        scene.add(Transmitter(
            name="hidden-source",
            position=mi.Point3f(source_xy_m[0], source_xy_m[1], self.source_altitude_m),
            power_dbm=float(source_power_dbm),
        ))
        west, east, south, north = spec.bounds_enu_m
        radio_map = RadioMapSolver()(
            scene,
            center=mi.Point3f((west + east) / 2.0, (south + north) / 2.0, spec.receiver_altitude_m),
            orientation=mi.Point3f(0.0, 0.0, 0.0),
            size=mi.Point2f(east - west, north - south),
            cell_size=mi.Point2f(*spec.cell_size_m),
            samples_per_tx=self.samples_per_tx,
            max_depth=self.max_depth,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=True,
            diffraction=False,
            seed=int(seed),
        )
        watts = np.asarray(radio_map.rss.numpy(), dtype=np.float64)
        if watts.ndim == 3 and watts.shape[0] == 1:
            watts = watts[0]
        if watts.shape != (spec.rows, spec.columns):
            raise ValueError(f"Sionna RadioMap 返回了未知形状: {watts.shape}")
        # Sionna rows progress south-to-north; persisted HWF row zero is north.
        watts = np.flip(watts, axis=0)
        return np.clip(10.0 * np.log10(np.maximum(watts, 1e-17) * 1_000.0), -140.0, 20.0).astype(np.float32)


def _spec_from_config(config: dict[str, Any]) -> MultibandRemSpec:
    shape = config.get("shape", [128, 128, 5])
    frequencies = tuple(float(value) for value in config.get("frequencies_hz", DEFAULT_REM_SPEC.frequencies_hz))
    if str(config.get("schema_version", SCHEMA_VERSION)) != SCHEMA_VERSION:
        raise ValueError(f"当前数据生成器只支持 schema_version={SCHEMA_VERSION}")
    if list(shape) != [128, 128, 5]:
        raise ValueError("当前多频 REM 数据契约固定为 [128,128,5]")
    return MultibandRemSpec(
        rows=int(shape[0]),
        columns=int(shape[1]),
        frequencies_hz=frequencies,
        receiver_altitude_m=float(config.get("receiver_altitude_m", 21.0)),
        bounds_enu_m=tuple(float(value) for value in config.get("bounds_enu_m", [-60, 60, -60, 60])),
        axis_order=str(config.get("axis_order", AXIS_ORDER)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/uav_rem_dataset.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scenes", type=int)
    parser.add_argument("--masks-per-scene", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--provider", choices=("sionna", "analytic"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    spec = _spec_from_config(config)
    dataset = config.get("dataset", {})
    ray_tracing = config.get("ray_tracing", {})
    output = args.output or Path(dataset.get("output", "data/uav-rem-multiband"))
    scene_count = int(args.scenes if args.scenes is not None else dataset.get("scenes", 1))
    masks_per_scene = int(args.masks_per_scene if args.masks_per_scene is not None else dataset.get("masks_per_scene", 1))
    base_seed = int(args.seed if args.seed is not None else dataset.get("seed", 20260812))
    provider_name = args.provider or str(config.get("provider", "sionna"))
    if scene_count <= 0 or masks_per_scene <= 0:
        raise ValueError("scenes 和 masks_per_scene 必须为正数")
    provider: DenseBandMapProvider
    if provider_name == "analytic":
        provider = AnalyticDenseMapProvider()
    else:
        provider = SionnaRadioMapProvider(
            asset_dir=ray_tracing.get("asset_dir", "/workspace/YiFan/spectrumclaw_runtime/artifacts/uav-rem-dataset/scene"),
            source_altitude_m=float(ray_tracing.get("source_altitude_m", 1.5)),
            samples_per_tx=int(ray_tracing.get("samples_per_tx", 10_000_000)),
            max_depth=int(ray_tracing.get("max_depth", 2)),
        )

    sampling = config.get("sampling", {})
    modes = list(sampling.get("modes", ["random_walk", "lawnmower", "random_points"]))
    ratio_range = sampling.get("ratio_range", [0.005, 0.05])
    manifest = {"spec": spec.metadata(), "provider": provider_name, "samples": []}
    for scene_offset in range(scene_count):
        scene_seed = base_seed + scene_offset
        dense = generate_dense_rem(provider=provider, spec=spec, seed=scene_seed)
        split = str(np.asarray(dense["split"]).item())
        for mask_index in range(masks_per_scene):
            mask_seed = scene_seed * 10_000 + mask_index
            rng = np.random.default_rng(mask_seed)
            mode = modes[mask_index % len(modes)]
            ratio = float(rng.uniform(float(ratio_range[0]), float(ratio_range[1])))
            sample = sample_dense_rem(
                dense,
                spec=spec,
                seed=mask_seed,
                sampling_ratio=ratio,
                trajectory_mode=mode,
            )
            filename = f"scene_{scene_seed:08d}_mask_{mask_index:03d}.npz"
            path = save_rem_sample(output / split / filename, sample, spec)
            manifest["samples"].append({
                "path": str(path.relative_to(output)),
                "split": split,
                "scene_seed": scene_seed,
                "mask_seed": mask_seed,
                "sampling_mode": mode,
                "sampling_ratio": float(np.asarray(sample["sampling_ratio"]).item()),
                "requested_sampling_ratio": ratio,
                "trajectory_points": int(len(sample["trajectory_rc"])),
                "unique_observation_points": int(np.count_nonzero(sample["observed_mask"][:, :, 0])),
            })
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "samples": len(manifest["samples"]), "provider": provider_name}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
