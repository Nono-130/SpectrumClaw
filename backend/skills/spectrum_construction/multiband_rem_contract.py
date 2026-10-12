"""Versioned contract for fixed-altitude, multi-band spectrum REM samples.

The persisted tensor order is always HWF: north-to-south rows, west-to-east
columns, then frequency bands.  Flight altitude is metadata, never a tensor
axis or a localization target.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .spatial_grid import (
    SPECTRUM_GRID_BOUNDS_ENU_M,
    SPECTRUM_GRID_COLUMNS,
    SPECTRUM_GRID_ROWS,
    SPECTRUM_RECEIVER_ALTITUDE_M,
)


SCHEMA_VERSION = "spectrumclaw.rem.hwf.v1"
AXIS_ORDER = "HWF"
DEFAULT_FREQUENCIES_HZ = (1.0e9, 1.4e9, 2.4e9, 3.5e9, 5.8e9)
_FORBIDDEN_HEIGHT_LABELS = {
    "source_z_m",
    "source_xyz_m",
    "emitter_z_m",
    "emitter_xyz_m",
    "emitter_height_m",
    "target_z_m",
    "target_xyz_m",
    "tx_z_m",
    "tx_positions",
    "height_map",
    "height_levels",
    "heights",
    "height_layer",
    "height_count",
}
_SPLIT_NAMES = frozenset(("train", "validation", "test"))
_SAMPLING_MODES = frozenset(("random_walk", "lawnmower", "random_points"))
_ALLOWED_SAMPLE_FIELDS = frozenset({
    "schema_version",
    "axis_order",
    "rss_dbm",
    "sampled_rss_dbm",
    "observed_mask",
    "frequencies_hz",
    "building_mask",
    "flyable_mask",
    "valid_rx_mask",
    "bounds_enu_m",
    "cell_size_m",
    "receiver_altitude_m",
    "source_xy_m",
    "source_power_dbm",
    "source_band_mask",
    "trajectory_xy_m",
    "trajectory_rc",
    "measurements_dbm",
    "scene_id",
    "seed",
    "split_group_id",
    "split",
    "sampling_ratio",
    "requested_sampling_ratio",
    "sampling_mode",
    "mask_seed",
})


@dataclass(frozen=True)
class MultibandRemSpec:
    """Authoritative spatial and spectral axes for one REM dataset."""

    rows: int = SPECTRUM_GRID_ROWS
    columns: int = SPECTRUM_GRID_COLUMNS
    frequencies_hz: tuple[float, ...] = DEFAULT_FREQUENCIES_HZ
    receiver_altitude_m: float = SPECTRUM_RECEIVER_ALTITUDE_M
    bounds_enu_m: tuple[float, float, float, float] = SPECTRUM_GRID_BOUNDS_ENU_M
    schema_version: str = SCHEMA_VERSION
    axis_order: str = AXIS_ORDER

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"v1 REM schema_version 必须固定为 {SCHEMA_VERSION}")
        if self.rows != 128 or self.columns != 128:
            raise ValueError("v1 REM 空间网格必须固定为 128×128")
        if self.axis_order != AXIS_ORDER:
            raise ValueError("REM 数据轴顺序必须是 HWF")
        if len(self.frequencies_hz) != 5:
            raise ValueError("REM 频谱轴必须恰好包含五个频段")
        frequencies = tuple(float(value) for value in self.frequencies_hz)
        if not all(math.isfinite(value) and value > 0 for value in frequencies):
            raise ValueError("频段中心频率必须为有限正数")
        if tuple(sorted(set(frequencies))) != frequencies:
            raise ValueError("五个频段中心频率必须唯一且严格升序")
        if not math.isfinite(self.receiver_altitude_m) or not math.isclose(
            self.receiver_altitude_m,
            21.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("v1 REM 接收平面高度必须固定为 21 m")
        west, east, south, north = (float(value) for value in self.bounds_enu_m)
        if not west < east or not south < north:
            raise ValueError("ENU 空间边界无效")

    @property
    def band_count(self) -> int:
        return len(self.frequencies_hz)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.rows, self.columns, self.band_count

    @property
    def cell_size_m(self) -> tuple[float, float]:
        west, east, south, north = self.bounds_enu_m
        return (east - west) / self.columns, (north - south) / self.rows

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "axis_order": self.axis_order,
            "shape": list(self.shape),
            "spatial_grid": {
                "frame": "local-ENU",
                "row_order": "north_to_south",
                "column_order": "west_to_east",
                "bounds_enu_m": list(self.bounds_enu_m),
                "cell_size_m": list(self.cell_size_m),
            },
            "frequencies_hz": list(self.frequencies_hz),
            "receiver_altitude_m": self.receiver_altitude_m,
            "localization_target": "emitter_xy_m",
        }

    def enu_to_row_column(self, east_m: float, north_m: float) -> tuple[int, int]:
        east_value = float(east_m)
        north_value = float(north_m)
        west, east, south, north = self.bounds_enu_m
        if not (west <= east_value < east and south <= north_value < north):
            raise ValueError("ENU 坐标超出 REM 空间边界")
        cell_east, cell_north = self.cell_size_m
        column = int(math.floor((east_value - west) / cell_east))
        south_up_row = int(math.floor((north_value - south) / cell_north))
        row = self.rows - 1 - south_up_row
        return row, column

    def row_column_to_enu(self, row: int, column: int) -> tuple[float, float]:
        selected_row = int(row)
        selected_column = int(column)
        if not (0 <= selected_row < self.rows and 0 <= selected_column < self.columns):
            raise ValueError("REM 行列索引超出空间边界")
        west, _east, _south, north = self.bounds_enu_m
        cell_east, cell_north = self.cell_size_m
        return (
            west + (selected_column + 0.5) * cell_east,
            north - (selected_row + 0.5) * cell_north,
        )


DEFAULT_REM_SPEC = MultibandRemSpec()


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError("标量元数据形状无效")
    return array.item()


def _require_shape(sample: Mapping[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray:
    if key not in sample:
        raise ValueError(f"REM 样本缺少字段: {key}")
    value = np.asarray(sample[key])
    if value.shape != shape:
        raise ValueError(f"REM 字段 {key} 形状应为 {shape}，实际为 {value.shape}")
    return value


def _require_binary(name: str, value: np.ndarray) -> None:
    if not np.all((value == 0) | (value == 1)):
        raise ValueError(f"REM 字段 {name} 必须是二值掩码")


def validate_rem_sample(
    sample: Mapping[str, Any],
    spec: MultibandRemSpec = DEFAULT_REM_SPEC,
) -> None:
    """Validate a persisted sample without silently accepting legacy ZHW data."""
    leaked_labels = sorted(_FORBIDDEN_HEIGHT_LABELS.intersection(sample))
    if leaked_labels:
        raise ValueError(f"XY-only 定位契约禁止高度标签: {', '.join(leaked_labels)}")
    undeclared_fields = sorted(set(sample).difference(_ALLOWED_SAMPLE_FIELDS))
    if undeclared_fields:
        raise ValueError(
            f"v1 REM schema 含未声明字段: {', '.join(undeclared_fields)}；新增语义必须升级 schema"
        )
    if _scalar(sample.get("schema_version")) != spec.schema_version:
        raise ValueError(f"REM schema_version 必须是 {spec.schema_version}")
    if _scalar(sample.get("axis_order")) != AXIS_ORDER:
        raise ValueError("REM 数据必须使用 HWF 轴顺序，不能按 ZHW 高度层解释")

    rss = _require_shape(sample, "rss_dbm", spec.shape).astype(np.float64, copy=False)
    sampled = _require_shape(sample, "sampled_rss_dbm", spec.shape).astype(np.float64, copy=False)
    observed = _require_shape(sample, "observed_mask", spec.shape)
    building = _require_shape(sample, "building_mask", (spec.rows, spec.columns))
    flyable = _require_shape(sample, "flyable_mask", (spec.rows, spec.columns))
    valid_rx = _require_shape(sample, "valid_rx_mask", (spec.rows, spec.columns))
    for name, value in (
        ("observed_mask", observed),
        ("building_mask", building),
        ("flyable_mask", flyable),
        ("valid_rx_mask", valid_rx),
    ):
        _require_binary(name, value)
    if np.any((flyable == 1) & (valid_rx == 0)):
        raise ValueError("flyable_mask 必须是 valid_rx_mask 的子集")

    if not np.all(np.isfinite(rss)):
        raise ValueError("rss_dbm 必须包含有限的五频段稠密真值")
    if np.any(np.isfinite(sampled[observed == 0])):
        raise ValueError("sampled_rss_dbm 未观测位置必须为 NaN，并与 observed_mask 一致")
    if np.any(~np.isfinite(sampled[observed == 1])):
        raise ValueError("sampled_rss_dbm 已观测位置必须有限，并与 observed_mask 一致")

    frequencies = _require_shape(sample, "frequencies_hz", (spec.band_count,)).astype(float)
    if not np.array_equal(frequencies, np.asarray(spec.frequencies_hz, dtype=float)):
        raise ValueError("frequencies_hz 与权威五频段顺序不一致")
    bounds = _require_shape(sample, "bounds_enu_m", (4,)).astype(float)
    if not np.allclose(bounds, spec.bounds_enu_m, rtol=0.0, atol=1e-6):
        raise ValueError("bounds_enu_m 与数据集空间网格不一致")
    cell_size = _require_shape(sample, "cell_size_m", (2,)).astype(float)
    if not np.allclose(cell_size, spec.cell_size_m, rtol=0.0, atol=1e-6):
        raise ValueError("cell_size_m 与 128×128 空间网格不一致")
    if not math.isclose(float(_scalar(sample.get("receiver_altitude_m"))), spec.receiver_altitude_m, abs_tol=1e-6):
        raise ValueError("receiver_altitude_m 必须保持固定单高度平面")

    source_xy = _require_shape(sample, "source_xy_m", (1, 2)).astype(float)
    spec.enu_to_row_column(float(source_xy[0, 0]), float(source_xy[0, 1]))
    _require_shape(sample, "source_power_dbm", (1, spec.band_count))
    source_band_mask = _require_shape(sample, "source_band_mask", (1, spec.band_count))
    _require_binary("source_band_mask", source_band_mask)

    trajectory_rc = np.asarray(sample.get("trajectory_rc"))
    trajectory_xy = np.asarray(sample.get("trajectory_xy_m"))
    measurements = np.asarray(sample.get("measurements_dbm"))
    if trajectory_rc.ndim != 2 or trajectory_rc.shape[1:] != (2,):
        raise ValueError("trajectory_rc 必须是 [N,2]")
    if not np.issubdtype(trajectory_rc.dtype, np.integer):
        raise ValueError("trajectory_rc 必须使用整数行列索引")
    if trajectory_xy.shape != trajectory_rc.shape:
        raise ValueError("trajectory_xy_m 必须与 trajectory_rc 一一对应")
    if measurements.shape != (trajectory_rc.shape[0], spec.band_count):
        raise ValueError("measurements_dbm 必须是 [N,5]")
    if not np.all(np.isfinite(trajectory_xy)):
        raise ValueError("trajectory_xy_m 必须全部为有限坐标")
    if not np.all(np.isfinite(measurements)):
        raise ValueError("measurements_dbm 必须全部为有限测量值")
    if trajectory_rc.size:
        if np.any(trajectory_rc[:, 0] < 0) or np.any(trajectory_rc[:, 0] >= spec.rows):
            raise ValueError("trajectory_rc 行索引越界")
        if np.any(trajectory_rc[:, 1] < 0) or np.any(trajectory_rc[:, 1] >= spec.columns):
            raise ValueError("trajectory_rc 列索引越界")
        trajectory_rows = trajectory_rc[:, 0]
        trajectory_columns = trajectory_rc[:, 1]
        if np.any(flyable[trajectory_rows, trajectory_columns] != 1):
            raise ValueError("轨迹采样点必须全部位于 flyable_mask 可飞区域")
        if np.any(valid_rx[trajectory_rows, trajectory_columns] != 1):
            raise ValueError("轨迹采样点必须全部位于 valid_rx_mask 有效接收区域")
        west, _east, _south, north = spec.bounds_enu_m
        cell_east, cell_north = spec.cell_size_m
        expected_xy = np.column_stack((
            west + (trajectory_rc[:, 1] + 0.5) * cell_east,
            north - (trajectory_rc[:, 0] + 0.5) * cell_north,
        ))
        if not np.allclose(trajectory_xy, expected_xy, rtol=0.0, atol=1e-5):
            raise ValueError("trajectory_xy_m 与 trajectory_rc 的 128×128 网格中心不一致")
        expected_measurements = rss[trajectory_rc[:, 0], trajectory_rc[:, 1], :]
        if not np.allclose(measurements, expected_measurements, rtol=0.0, atol=1e-5):
            raise ValueError("measurements_dbm 与 trajectory_rc 对应的稠密 RSS 不一致")

    expected_observed = np.zeros(spec.shape, dtype=np.uint8)
    if trajectory_rc.size:
        expected_observed[trajectory_rc[:, 0], trajectory_rc[:, 1], :] = 1
    if not np.array_equal(observed, expected_observed):
        raise ValueError("observed_mask 必须与 trajectory_rc 的五频同步观测严格一致")
    if not np.allclose(sampled[observed == 1], rss[observed == 1], rtol=0.0, atol=1e-5):
        raise ValueError("sampled_rss_dbm 必须等于轨迹位置的稠密 RSS 真值")

    for scalar_key in (
        "scene_id",
        "seed",
        "split_group_id",
        "split",
        "sampling_ratio",
        "requested_sampling_ratio",
        "sampling_mode",
        "mask_seed",
    ):
        if scalar_key not in sample:
            raise ValueError(f"REM 样本缺少字段: {scalar_key}")
        _scalar(sample[scalar_key])
    split_group_id = str(_scalar(sample["split_group_id"]))
    split = str(_scalar(sample["split"]))
    expected_split = assign_split(split_group_id, seed=0)
    if split not in _SPLIT_NAMES or split != expected_split:
        raise ValueError(
            f"split 必须与 split_group_id 的确定性分组一致，应为 {expected_split}"
        )
    sampling_mode = str(_scalar(sample["sampling_mode"]))
    if sampling_mode not in _SAMPLING_MODES:
        raise ValueError(f"sampling_mode 不受支持: {sampling_mode}")
    requested_sampling_ratio = float(_scalar(sample["requested_sampling_ratio"]))
    if not math.isfinite(requested_sampling_ratio) or not 0 < requested_sampling_ratio <= 1:
        raise ValueError("requested_sampling_ratio 必须位于 (0,1]")
    sampling_ratio = float(_scalar(sample["sampling_ratio"]))
    unique_observations = int(np.count_nonzero(observed[:, :, 0]))
    expected_sampling_ratio = unique_observations / float(spec.rows * spec.columns)
    if not math.isclose(sampling_ratio, expected_sampling_ratio, rel_tol=0.0, abs_tol=1e-7):
        raise ValueError("sampling_ratio 必须等于 observed_mask 的实际空间覆盖率")


def _validate_split_path(path: str | Path, sample: Mapping[str, Any]) -> None:
    directory_split = Path(path).parent.name
    if directory_split not in _SPLIT_NAMES:
        return
    sample_split = str(_scalar(sample.get("split")))
    if directory_split != sample_split:
        raise ValueError(
            f"REM 样本 split={sample_split} 与目标目录 {directory_split}/ 不一致"
        )


def save_rem_sample(
    path: str | Path,
    sample: Mapping[str, Any],
    spec: MultibandRemSpec = DEFAULT_REM_SPEC,
) -> Path:
    validate_rem_sample(sample, spec)
    target = Path(path)
    _validate_split_path(target, sample)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".npz", dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **sample)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_rem_sample(
    path: str | Path,
    spec: MultibandRemSpec = DEFAULT_REM_SPEC,
) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        sample = {key: archive[key] for key in archive.files}
    validate_rem_sample(sample, spec)
    _validate_split_path(path, sample)
    return sample


def assign_split(
    split_group_id: str,
    *,
    seed: int = 0,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> str:
    """Assign an entire dense/source group to one stable split."""
    if not 0 < train_ratio < 1 or not 0 <= validation_ratio < 1 - train_ratio:
        raise ValueError("训练/验证数据比例无效")
    digest = hashlib.blake2b(
        f"{int(seed)}:{split_group_id}".encode("utf-8"),
        digest_size=8,
    ).digest()
    unit_value = int.from_bytes(digest, "big") / float(1 << 64)
    if unit_value < train_ratio:
        return "train"
    if unit_value < train_ratio + validation_ratio:
        return "validation"
    return "test"
