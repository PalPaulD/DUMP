"""
Minimal tests for w0waCDM upweighting functionality.
Run with: pytest DUMP/tests/test_w0wacdm_upweighting.py -v
"""
import torch
import numpy as np


def test_weighted_loss_computation():
    """Test that weighted loss correctly applies different weights to ΛCDM vs w0waCDM samples."""
    batch_size = 4
    n_z = 3
    n_k = 10

    # Mixed batch: [ΛCDM, w0waCDM, ΛCDM, w0waCDM]
    cosmo = {'is_w0wacdm': torch.tensor([0.0, 1.0, 0.0, 1.0])}

    # Dummy predictions and targets
    target_pred = torch.randn(batch_size, n_z, n_k)
    target = torch.randn(batch_size, n_z, n_k)

    # Apply weighting (mimics model.training_step logic)
    w0wacdm_loss_weight = 5.0
    weights = torch.where(cosmo.get('is_w0wacdm', 0) > 0.5, w0wacdm_loss_weight, 1.0)

    # Check weights are correct
    expected_weights = torch.tensor([1.0, 5.0, 1.0, 5.0])
    assert torch.allclose(weights, expected_weights), f"Expected {expected_weights}, got {weights}"

    # Compute weighted loss
    mse_per_sample = torch.nn.functional.mse_loss(target_pred, target, reduction='none').mean(dim=(1, 2))
    weighted_loss = (weights * mse_per_sample).mean()

    # Verify it's different from unweighted
    unweighted_loss = mse_per_sample.mean()
    assert weighted_loss != unweighted_loss, "Weighted loss should differ from unweighted"

    print(f"Weighted loss test passed (weighted: {weighted_loss:.4f}, unweighted: {unweighted_loss:.4f})")


def test_sampling_distribution():
    """Test that WeightedRandomSampler oversamples w0waCDM."""
    from torch.utils.data import WeightedRandomSampler

    n_lcdm = 100
    n_w0wacdm = 10
    sampling_freq = 50  # Sample w0waCDM 50× more

    # Create weights
    sample_weights = [1.0] * n_lcdm + [sampling_freq] * n_w0wacdm
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=2000, replacement=True)

    # Sample and count
    samples = list(sampler)
    lcdm_count = sum(1 for idx in samples if idx < n_lcdm)
    w0wacdm_count = sum(1 for idx in samples if idx >= n_lcdm)

    # Expected ratio calculation:
    # Total weight: 100*1.0 + 10*50 = 600
    # P(w0waCDM) = 500/600, P(ΛCDM) = 100/600
    # Ratio = 500/100 = 5
    expected_ratio = (n_w0wacdm * sampling_freq) / (n_lcdm * 1.0)

    observed_ratio = w0wacdm_count / lcdm_count if lcdm_count > 0 else 0
    assert 0.8 * expected_ratio < observed_ratio < 1.2 * expected_ratio, \
        f"Expected ratio ~{expected_ratio:.1f}, got {observed_ratio:.1f} (ΛCDM: {lcdm_count}, w0waCDM: {w0wacdm_count})"

    print(f"Sampling distribution test passed (ratio: {observed_ratio:.1f}, expected: ~{expected_ratio:.1f})")


def test_loss_weight_one_equals_no_weighting():
    """Test that w0wacdm_loss_weight=1.0 is equivalent to no weighting."""
    batch_size = 4
    n_z = 3
    n_k = 10

    cosmo = {'is_w0wacdm': torch.tensor([0.0, 1.0, 0.0, 1.0])}
    target_pred = torch.randn(batch_size, n_z, n_k)
    target = torch.randn(batch_size, n_z, n_k)

    # Weight = 1.0 for both
    weights = torch.where(cosmo.get('is_w0wacdm', 0) > 0.5, 1.0, 1.0)
    weighted_loss = (weights * torch.nn.functional.mse_loss(target_pred, target, reduction='none').mean(dim=(1, 2))).mean()

    # Unweighted
    unweighted_loss = torch.nn.functional.mse_loss(target_pred, target)

    assert torch.allclose(weighted_loss, unweighted_loss), \
        "Loss with weight=1.0 should equal unweighted loss"

    print(f"Weight=1.0 equivalence test passed")


def test_missing_flag_defaults_to_lcdm():
    """Test that missing is_w0wacdm flag defaults to ΛCDM (weight=1.0)."""
    cosmo = {}  # No is_w0wacdm key

    # Get default value (should be 0, treated as tensor)
    is_w0wacdm = torch.tensor(cosmo.get('is_w0wacdm', 0.0))

    w0wacdm_loss_weight = 10.0
    weights = torch.where(is_w0wacdm > 0.5, w0wacdm_loss_weight, 1.0)

    assert weights == 1.0, "Missing flag should default to weight=1.0 (ΛCDM)"

    print(f"✓ Missing flag default test passed")


if __name__ == "__main__":
    print("\nRunning w0waCDM upweighting tests...\n")
    test_weighted_loss_computation()
    test_sampling_distribution()
    test_loss_weight_one_equals_no_weighting()
    test_missing_flag_defaults_to_lcdm()
    print("\nAll tests passed!\n")
