"""Torch contract tests for joint five-band REM reconstruction and XY localization."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")


def _inputs(size=32):
    sparse = torch.full((1, size, size, 5), -100.0)
    mask = torch.zeros_like(sparse)
    prior = torch.full_like(sparse, -85.0)
    building = torch.zeros((1, size, size))
    sparse[:, 4, 7, :] = -52.0
    mask[:, 4, 7, :] = 1.0
    frequencies = torch.tensor([[1.0e9, 1.4e9, 2.4e9, 3.5e9, 5.8e9]])
    return sparse, mask, prior, building, frequencies


def test_model_returns_five_rem_bands_uncertainty_and_xy_without_z():
    from backend.skills.spectrum_construction.multiband_rem_model import MultibandResidualUNet

    model = MultibandResidualUNet(num_bands=5, base_channels=8, bounds_enu_m=(-60, 60, -60, 60))
    sparse, mask, prior, building, frequencies = _inputs()

    output = model(
        sparse_rss_dbm=sparse,
        observed_mask=mask,
        physics_prior_dbm=prior,
        building_mask=building,
        frequencies_hz=frequencies,
    )

    assert output["rem_mean_dbm"].shape == (1, 32, 32, 5)
    assert output["rem_log_variance"].shape == (1, 32, 32, 5)
    assert output["emitter_xy_m"].shape == (1, 2)
    assert output["emitter_covariance_xy"].shape == (1, 2, 2)
    assert "emitter_z_m" not in output


def test_model_hard_preserves_measured_rss_values():
    from backend.skills.spectrum_construction.multiband_rem_model import MultibandResidualUNet

    model = MultibandResidualUNet(num_bands=5, base_channels=8, bounds_enu_m=(-60, 60, -60, 60))
    sparse, mask, prior, building, frequencies = _inputs()

    output = model(
        sparse_rss_dbm=sparse,
        observed_mask=mask,
        physics_prior_dbm=prior,
        building_mask=building,
        frequencies_hz=frequencies,
    )

    torch.testing.assert_close(output["rem_mean_dbm"][:, 4, 7, :], sparse[:, 4, 7, :])


def test_joint_training_loss_is_finite_for_rem_and_xy_targets():
    from backend.skills.spectrum_construction.multiband_rem_model import (
        MultibandResidualUNet,
        multiband_rem_loss,
    )

    model = MultibandResidualUNet(num_bands=5, base_channels=8, bounds_enu_m=(-60, 60, -60, 60))
    sparse, mask, prior, building, frequencies = _inputs()
    output = model(
        sparse_rss_dbm=sparse,
        observed_mask=mask,
        physics_prior_dbm=prior,
        building_mask=building,
        frequencies_hz=frequencies,
    )
    target = prior + torch.randn_like(prior)
    source_xy = torch.tensor([[12.0, -8.0]])

    losses = multiband_rem_loss(
        output,
        target_rem_dbm=target,
        source_xy_m=source_xy,
        observed_mask=mask,
    )

    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(losses["rem_nll"])
    assert torch.isfinite(losses["xy_smooth_l1"])


def test_rem_loss_ignores_nan_targets_outside_valid_receiver_mask():
    from backend.skills.spectrum_construction.multiband_rem_model import (
        MultibandResidualUNet,
        multiband_rem_loss,
    )

    model = MultibandResidualUNet(num_bands=5, base_channels=8, bounds_enu_m=(-60, 60, -60, 60))
    sparse, mask, prior, building, frequencies = _inputs()
    output = model(
        sparse_rss_dbm=sparse,
        observed_mask=mask,
        physics_prior_dbm=prior,
        building_mask=building,
        frequencies_hz=frequencies,
    )
    target = prior.clone()
    target[:, 2, 3, :] = torch.nan
    valid = torch.ones((1, 32, 32))
    valid[:, 2, 3] = 0.0

    losses = multiband_rem_loss(
        output,
        target_rem_dbm=target,
        source_xy_m=torch.tensor([[12.0, -8.0]]),
        observed_mask=mask,
        valid_mask=valid,
    )

    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(losses["rem_nll"])


def test_subcell_offset_changes_xy_without_inflating_softmax_covariance():
    from backend.skills.spectrum_construction.multiband_rem_model import _soft_argmax_xy

    logits = torch.zeros((1, 8, 8))
    _, covariance_without_offset = _soft_argmax_xy(
        logits,
        bounds_enu_m=(-60.0, 60.0, -60.0, 60.0),
        subcell_offset=torch.zeros((1, 2)),
    )
    shifted_xy, covariance_with_offset = _soft_argmax_xy(
        logits,
        bounds_enu_m=(-60.0, 60.0, -60.0, 60.0),
        subcell_offset=torch.tensor([[0.75, -0.5]]),
    )

    assert not torch.allclose(shifted_xy, torch.zeros_like(shifted_xy))
    torch.testing.assert_close(covariance_with_offset, covariance_without_offset)
