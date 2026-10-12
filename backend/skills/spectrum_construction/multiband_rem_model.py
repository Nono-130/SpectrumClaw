"""Probabilistic residual U-Net for five-band REM and emitter XY estimation.

This module intentionally lives outside the main backend dependency surface.
Training and inference use the dedicated REM environment declared in
``config/dependencies/requirements-rem-training.txt``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class _ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, *, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


def _group_count(channels: int) -> int:
    for candidate in (8, 4, 2):
        if channels % candidate == 0:
            return candidate
    return 1


class _UpBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int, *, dropout: float) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2)
        self.conv = _ConvBlock(output_channels + skip_channels, output_channels, dropout=dropout)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = self.up(value)
        if value.shape[-2:] != skip.shape[-2:]:
            value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat((value, skip), dim=1))


class MultibandResidualUNet(nn.Module):
    """Reconstruct five REM bands and infer one shared continuous XY source."""

    def __init__(
        self,
        *,
        num_bands: int = 5,
        base_channels: int = 32,
        bounds_enu_m: tuple[float, float, float, float] = (-60.0, 60.0, -60.0, 60.0),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if num_bands != 5:
            raise ValueError("当前 REM 模型契约固定为五个频段")
        self.num_bands = num_bands
        self.bounds_enu_m = tuple(float(value) for value in bounds_enu_m)
        input_channels = num_bands * 4 + 3
        self.encoder1 = _ConvBlock(input_channels, base_channels, dropout=dropout)
        self.encoder2 = _ConvBlock(base_channels, base_channels * 2, dropout=dropout)
        self.encoder3 = _ConvBlock(base_channels * 2, base_channels * 4, dropout=dropout)
        self.encoder4 = _ConvBlock(base_channels * 4, base_channels * 8, dropout=dropout)
        self.bottleneck = _ConvBlock(base_channels * 8, base_channels * 16, dropout=dropout)
        self.pool = nn.MaxPool2d(2)
        self.up4 = _UpBlock(base_channels * 16, base_channels * 8, base_channels * 8, dropout=dropout)
        self.up3 = _UpBlock(base_channels * 8, base_channels * 4, base_channels * 4, dropout=dropout)
        self.up2 = _UpBlock(base_channels * 4, base_channels * 2, base_channels * 2, dropout=dropout)
        self.up1 = _UpBlock(base_channels * 2, base_channels, base_channels, dropout=dropout)
        self.residual_head = nn.Conv2d(base_channels, num_bands, kernel_size=1)
        self.log_variance_head = nn.Conv2d(base_channels, num_bands, kernel_size=1)
        self.localization_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.offset_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 16, 2),
            nn.Tanh(),
        )

    def forward(
        self,
        *,
        sparse_rss_dbm: torch.Tensor,
        observed_mask: torch.Tensor,
        physics_prior_dbm: torch.Tensor,
        building_mask: torch.Tensor,
        frequencies_hz: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if sparse_rss_dbm.ndim != 4 or sparse_rss_dbm.shape[-1] != self.num_bands:
            raise ValueError("sparse_rss_dbm 必须使用 [B,H,W,5] 轴顺序")
        if observed_mask.shape != sparse_rss_dbm.shape or physics_prior_dbm.shape != sparse_rss_dbm.shape:
            raise ValueError("稀疏 RSS、观测掩码和物理先验必须具有相同的 BHWF 形状")
        batch, rows, columns, _bands = sparse_rss_dbm.shape
        if building_mask.shape == (batch, rows, columns):
            building = building_mask.unsqueeze(1)
        elif building_mask.shape == (batch, 1, rows, columns):
            building = building_mask
        else:
            raise ValueError("building_mask 必须是 [B,H,W] 或 [B,1,H,W]")
        if frequencies_hz.ndim == 1:
            frequencies_hz = frequencies_hz.unsqueeze(0).expand(batch, -1)
        if frequencies_hz.shape != (batch, self.num_bands):
            raise ValueError("frequencies_hz 必须是 [B,5] 或 [5]")

        sparse = sparse_rss_dbm.permute(0, 3, 1, 2).float()
        mask = observed_mask.permute(0, 3, 1, 2).float()
        prior = physics_prior_dbm.permute(0, 3, 1, 2).float()
        sparse_filled = torch.where(mask > 0.5, sparse, prior)
        band_planes = torch.log10(frequencies_hz.float().clamp_min(1.0) / 1.0e9)
        band_planes = band_planes[:, :, None, None].expand(-1, -1, rows, columns)
        coordinate_planes = _normalized_coordinate_planes(
            batch=batch,
            rows=rows,
            columns=columns,
            device=sparse.device,
            dtype=sparse.dtype,
        )
        value = torch.cat(
            (sparse_filled, mask, prior, band_planes, building.float(), coordinate_planes),
            dim=1,
        )

        encoder1 = self.encoder1(value)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        encoder4 = self.encoder4(self.pool(encoder3))
        bottleneck = self.bottleneck(self.pool(encoder4))
        decoded = self.up4(bottleneck, encoder4)
        decoded = self.up3(decoded, encoder3)
        decoded = self.up2(decoded, encoder2)
        decoded = self.up1(decoded, encoder1)

        residual = self.residual_head(decoded)
        rem_mean = prior + residual
        rem_mean = torch.where(mask > 0.5, sparse, rem_mean)
        rem_log_variance = self.log_variance_head(decoded).clamp(-6.0, 6.0)
        localization_logits = self.localization_head(decoded).squeeze(1)
        emitter_xy, covariance = _soft_argmax_xy(
            localization_logits,
            bounds_enu_m=self.bounds_enu_m,
            subcell_offset=self.offset_head(bottleneck),
        )
        return {
            "rem_mean_dbm": rem_mean.permute(0, 2, 3, 1),
            "rem_log_variance": rem_log_variance.permute(0, 2, 3, 1),
            "localization_logits": localization_logits,
            "emitter_xy_m": emitter_xy,
            "emitter_covariance_xy": covariance,
            "bounds_enu_m": torch.as_tensor(
                self.bounds_enu_m,
                device=rem_mean.device,
                dtype=rem_mean.dtype,
            ),
        }


def _normalized_coordinate_planes(
    *,
    batch: int,
    rows: int,
    columns: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    north_to_south = torch.linspace(1.0, -1.0, rows, device=device, dtype=dtype)
    west_to_east = torch.linspace(-1.0, 1.0, columns, device=device, dtype=dtype)
    north, east = torch.meshgrid(north_to_south, west_to_east, indexing="ij")
    return torch.stack((east, north), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)


def _soft_argmax_xy(
    logits: torch.Tensor,
    *,
    bounds_enu_m: tuple[float, float, float, float],
    subcell_offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, rows, columns = logits.shape
    west, east, south, north = bounds_enu_m
    cell_east = (east - west) / columns
    cell_north = (north - south) / rows
    east_axis = torch.linspace(
        west + cell_east / 2.0,
        east - cell_east / 2.0,
        columns,
        device=logits.device,
        dtype=logits.dtype,
    )
    north_axis = torch.linspace(
        north - cell_north / 2.0,
        south + cell_north / 2.0,
        rows,
        device=logits.device,
        dtype=logits.dtype,
    )
    north_grid, east_grid = torch.meshgrid(north_axis, east_axis, indexing="ij")
    probability = torch.softmax(logits.reshape(batch, -1), dim=-1).reshape(batch, rows, columns)
    expected_east = (probability * east_grid).sum(dim=(1, 2))
    expected_north = (probability * north_grid).sum(dim=(1, 2))
    offset_scale = logits.new_tensor((cell_east / 2.0, cell_north / 2.0))
    emitter_xy = torch.stack((expected_east, expected_north), dim=-1) + subcell_offset * offset_scale

    # The heatmap covariance describes the spatial probability distribution.
    # The separately regressed sub-cell offset moves the point estimate only;
    # centering covariance on that shifted point would add an artificial oo^T.
    delta_east = east_grid.unsqueeze(0) - expected_east[:, None, None]
    delta_north = north_grid.unsqueeze(0) - expected_north[:, None, None]
    covariance_ee = (probability * delta_east.square()).sum(dim=(1, 2))
    covariance_nn = (probability * delta_north.square()).sum(dim=(1, 2))
    covariance_en = (probability * delta_east * delta_north).sum(dim=(1, 2))
    covariance = torch.stack(
        (covariance_ee, covariance_en, covariance_en, covariance_nn),
        dim=-1,
    ).reshape(batch, 2, 2)
    covariance = covariance + torch.eye(2, device=logits.device, dtype=logits.dtype).unsqueeze(0) * 1e-6
    return emitter_xy, covariance


def multiband_rem_loss(
    output: dict[str, torch.Tensor],
    *,
    target_rem_dbm: torch.Tensor,
    source_xy_m: torch.Tensor,
    observed_mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    xy_weight: float = 0.20,
    heatmap_weight: float = 0.05,
) -> dict[str, torch.Tensor]:
    mean = output["rem_mean_dbm"]
    log_variance = output["rem_log_variance"]
    if mean.shape != target_rem_dbm.shape or observed_mask.shape != mean.shape:
        raise ValueError("REM loss 输入必须具有相同的 BHWF 形状")
    evaluation_mask = (observed_mask <= 0.5).to(mean.dtype)
    if valid_mask is not None:
        if valid_mask.ndim == 3:
            valid_mask = valid_mask.unsqueeze(-1)
        evaluation_mask = evaluation_mask * valid_mask.to(mean.dtype)
    active = evaluation_mask > 0.0
    if torch.any(active & ~torch.isfinite(target_rem_dbm)):
        raise ValueError("有效接收区域的 target_rem_dbm 必须为有限值")
    safe_target = torch.where(active, target_rem_dbm, mean)
    squared_error = (mean - safe_target).square()
    nll_per_cell = 0.5 * (torch.exp(-log_variance) * squared_error + log_variance)
    denominator = evaluation_mask.sum().clamp_min(1.0)
    rem_nll = torch.where(active, nll_per_cell * evaluation_mask, torch.zeros_like(nll_per_cell)).sum() / denominator
    xy_smooth_l1 = F.smooth_l1_loss(output["emitter_xy_m"], source_xy_m.float())

    bounds = output["bounds_enu_m"]
    west, east, south, north = (bounds[index] for index in range(4))
    rows, columns = output["localization_logits"].shape[-2:]
    target_column = torch.floor((source_xy_m[:, 0] - west) / (east - west) * columns)
    target_row = torch.floor((north - source_xy_m[:, 1]) / (north - south) * rows)
    target_column = target_column.long().clamp(0, columns - 1)
    target_row = target_row.long().clamp(0, rows - 1)
    target_index = target_row * columns + target_column
    heatmap_cross_entropy = F.cross_entropy(
        output["localization_logits"].reshape(source_xy_m.shape[0], -1),
        target_index,
    )
    total = rem_nll + float(xy_weight) * xy_smooth_l1 + float(heatmap_weight) * heatmap_cross_entropy
    return {
        "loss": total,
        "rem_nll": rem_nll,
        "xy_smooth_l1": xy_smooth_l1,
        "heatmap_cross_entropy": heatmap_cross_entropy,
    }


def checkpoint_metadata(model: MultibandResidualUNet) -> dict[str, Any]:
    return {
        "model_family": "probabilistic_residual_unet",
        "axis_order": "HWF",
        "num_bands": model.num_bands,
        "localization_target": "emitter_xy_m",
        "bounds_enu_m": list(model.bounds_enu_m),
    }
