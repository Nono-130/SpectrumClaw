#!/usr/bin/env python3
"""Train the fixed-altitude five-band REM reconstruction/localization model."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.skills.spectrum_construction.multiband_rem_contract import DEFAULT_REM_SPEC
from backend.skills.spectrum_construction.multiband_rem_model import (
    MultibandResidualUNet,
    checkpoint_metadata,
    multiband_rem_loss,
)
from backend.skills.spectrum_construction.multiband_rem_training import (
    MultibandRemDataset,
    discover_split_paths,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(
    *,
    model: MultibandResidualUNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "rem_nll": 0.0, "xy_smooth_l1": 0.0, "xy_error_m": 0.0}
    count = 0
    for batch in loader:
        tensors = {key: value.to(device) for key, value in batch.items()}
        with torch.set_grad_enabled(training):
            output = model(
                sparse_rss_dbm=tensors["sparse_rss_dbm"],
                observed_mask=tensors["observed_mask"],
                physics_prior_dbm=tensors["physics_prior_dbm"],
                building_mask=tensors["building_mask"],
                frequencies_hz=tensors["frequencies_hz"],
            )
            losses = multiband_rem_loss(
                output,
                target_rem_dbm=tensors["target_rem_dbm"],
                source_xy_m=tensors["source_xy_m"],
                observed_mask=tensors["observed_mask"],
                valid_mask=tensors["valid_rx_mask"],
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        batch_size = int(tensors["source_xy_m"].shape[0])
        count += batch_size
        totals["loss"] += float(losses["loss"].detach()) * batch_size
        totals["rem_nll"] += float(losses["rem_nll"].detach()) * batch_size
        totals["xy_smooth_l1"] += float(losses["xy_smooth_l1"].detach()) * batch_size
        totals["xy_error_m"] += float(
            torch.linalg.vector_norm(output["emitter_xy_m"] - tensors["source_xy_m"], dim=-1).sum().detach()
        )
    if count == 0:
        raise ValueError("训练/验证 split 中没有可用样本")
    return {key: value / count for key, value in totals.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/uav-rem-multiband"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--band-dropout", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs 和 batch-size 必须为正数")

    _seed_everything(args.seed)
    train_paths = discover_split_paths(args.data, "train")
    validation_paths = discover_split_paths(args.data, "validation")
    if not train_paths:
        raise ValueError(f"没有找到训练样本: {args.data / 'train'}")
    if not validation_paths:
        raise ValueError(f"没有找到验证样本: {args.data / 'validation'}")
    train_dataset = MultibandRemDataset(
        train_paths,
        band_dropout_probability=args.band_dropout,
        seed=args.seed,
    )
    validation_dataset = MultibandRemDataset(
        validation_paths,
        band_dropout_probability=0.0,
        seed=args.seed,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    device = torch.device(args.device)
    model = MultibandResidualUNet(
        num_bands=DEFAULT_REM_SPEC.band_count,
        base_channels=args.base_channels,
        bounds_enu_m=DEFAULT_REM_SPEC.bounds_enu_m,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    best_validation = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch - 1)
        train_metrics = _run_epoch(model=model, loader=train_loader, device=device, optimizer=optimizer)
        with torch.no_grad():
            validation_metrics = _run_epoch(model=model, loader=validation_loader, device=device, optimizer=None)
        record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if validation_metrics["loss"] < best_validation:
            best_validation = validation_metrics["loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation": validation_metrics,
                    "metadata": {
                        **checkpoint_metadata(model),
                        "schema_version": DEFAULT_REM_SPEC.schema_version,
                        "frequencies_hz": list(DEFAULT_REM_SPEC.frequencies_hz),
                        "receiver_altitude_m": DEFAULT_REM_SPEC.receiver_altitude_m,
                    },
                },
                args.output / "best_model.pt",
            )
        (args.output / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
