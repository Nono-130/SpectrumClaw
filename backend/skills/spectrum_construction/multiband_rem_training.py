"""Dataset utilities for the dedicated multi-band REM training environment."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from .multiband_rem_contract import DEFAULT_REM_SPEC, MultibandRemSpec, load_rem_sample


def build_idw_prior(
    sampled_rss_dbm: np.ndarray,
    observed_mask: np.ndarray,
    *,
    neighbors: int = 8,
    power: float = 2.0,
    fallback_dbm: float = -100.0,
) -> np.ndarray:
    """Build one independent spatial prior per frequency without label access."""
    sampled = np.asarray(sampled_rss_dbm, dtype=np.float32)
    observed = np.asarray(observed_mask)
    if sampled.ndim != 3 or sampled.shape != observed.shape:
        raise ValueError("IDW 输入必须是形状一致的 HWF 稀疏 RSS 和观测掩码")
    if neighbors <= 0 or power <= 0:
        raise ValueError("IDW 邻居数和幂指数必须为正数")
    rows, columns, bands = sampled.shape
    row_grid, column_grid = np.indices((rows, columns))
    query_points = np.column_stack((row_grid.ravel(), column_grid.ravel()))
    prior = np.empty_like(sampled)
    for band_index in range(bands):
        band_observed = observed[:, :, band_index] > 0
        points = np.argwhere(band_observed)
        if not len(points):
            prior[:, :, band_index] = float(fallback_dbm)
            continue
        values = sampled[:, :, band_index][band_observed]
        if not np.all(np.isfinite(values)):
            raise ValueError("IDW 已观测位置包含非有限 RSS")
        count = min(int(neighbors), len(points))
        distances, indices = cKDTree(points).query(query_points, k=count)
        if count == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        weights = 1.0 / np.maximum(distances, 1e-6) ** float(power)
        estimates = np.sum(weights * values[indices], axis=1) / np.sum(weights, axis=1)
        band_prior = estimates.reshape(rows, columns).astype(np.float32)
        band_prior[band_observed] = values
        prior[:, :, band_index] = band_prior
    return prior


class MultibandRemDataset(Dataset):
    """Load validated HWF samples and prepare finite tensors for the model."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        spec: MultibandRemSpec = DEFAULT_REM_SPEC,
        band_dropout_probability: float = 0.35,
        seed: int = 0,
    ) -> None:
        self.paths = tuple(Path(path) for path in paths)
        if not self.paths:
            raise ValueError("REM 训练数据集不能为空")
        if not 0.0 <= band_dropout_probability <= 1.0:
            raise ValueError("band_dropout_probability 必须位于 [0,1]")
        self.spec = spec
        self.band_dropout_probability = float(band_dropout_probability)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.paths)

    def set_epoch(self, epoch: int) -> None:
        """Select a reproducible but epoch-specific augmentation stream."""
        selected_epoch = int(epoch)
        if selected_epoch < 0:
            raise ValueError("epoch 不能为负数")
        self.epoch = selected_epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = load_rem_sample(self.paths[int(index)], self.spec)
        sampled = np.asarray(sample["sampled_rss_dbm"], dtype=np.float32).copy()
        observed = np.asarray(sample["observed_mask"], dtype=np.float32).copy()
        available = np.any(observed > 0.5, axis=(0, 1))
        rng = np.random.default_rng(np.random.SeedSequence((self.seed, self.epoch, int(index))))
        available_indices = np.flatnonzero(available)
        if len(available_indices) > 1 and rng.random() < self.band_dropout_probability:
            keep_count = int(rng.integers(1, len(available_indices)))
            keep = rng.choice(available_indices, size=keep_count, replace=False)
            available[:] = False
            available[keep] = True
        observed[:, :, ~available] = 0.0
        sampled[:, :, ~available] = np.nan
        prior = build_idw_prior(sampled, observed)
        sparse_finite = np.where(np.isfinite(sampled), sampled, prior).astype(np.float32)
        return {
            "sparse_rss_dbm": torch.from_numpy(sparse_finite),
            "observed_mask": torch.from_numpy(observed.astype(np.float32)),
            "physics_prior_dbm": torch.from_numpy(prior.astype(np.float32)),
            "target_rem_dbm": torch.from_numpy(np.asarray(sample["rss_dbm"], dtype=np.float32)),
            "building_mask": torch.from_numpy(np.asarray(sample["building_mask"], dtype=np.float32)),
            "valid_rx_mask": torch.from_numpy(np.asarray(sample["valid_rx_mask"], dtype=np.float32)),
            "frequencies_hz": torch.from_numpy(np.asarray(sample["frequencies_hz"], dtype=np.float32)),
            "band_available_mask": torch.from_numpy(available.astype(np.float32)),
            "source_xy_m": torch.from_numpy(np.asarray(sample["source_xy_m"], dtype=np.float32)[0]),
        }


def discover_split_paths(root: str | Path, split: str) -> list[Path]:
    selected = Path(root) / split
    return sorted(selected.glob("*.npz")) if selected.is_dir() else []
