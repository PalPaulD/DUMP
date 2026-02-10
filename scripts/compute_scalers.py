"""
Compute and save scalers for ALL features and targets.
Run this once before training. Models will filter what they need.
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from baccoemu import Matter_powerspectrum

from DUMP.data.features_engineering import make_features, nonlin_pk, func_map
from DUMP.data.constants import bacco_target_z, solver_dz
from DUMP.utils import find_solver_grid


def compute_scalers(
    bacco_emulator, 
    cosmologies_file: str, 
    target_z: np.ndarray,
    use_every_n: int = 1
):
    """Compute mean and std for ALL features and targets."""
    solver_z = find_solver_grid(target_z, solver_dz)

    cosmologies = pd.read_csv(cosmologies_file)
    n_samples = len(cosmologies)

    # Get first sample to determine shapes
    cosmo_0 = cosmologies.iloc[0].to_dict()
    features_0 = make_features(bacco_emulator, list(func_map.keys()), cosmo_0, solver_z)

    # Initialize accumulators
    feature_sums = {key: 0.0 for key, val in features_0.items()}
    feature_sq_sums = {key: 0.0 for key, val in features_0.items()}

    # Accumulators for lin_pk and nonlin_pk (single mean/std across all bins)
    target_sum = 0.0
    target_sq_sum = 0.0

    # Accumulate statistics
    n = 0
    for idx in tqdm(range(0, n_samples, use_every_n), desc="Estimating scalers...", unit="cosmo"):

        cosmo = cosmologies.iloc[idx].to_dict()

        features = make_features(bacco_emulator, list(func_map.keys()), cosmo, solver_z)
        for key in features:
            feature_sums[key] += features[key].mean()
            feature_sq_sums[key] += (features[key] ** 2).mean()

        # Accumulate for nonlin_pk (targets) - single mean/std across all bins and redshifts
        target = nonlin_pk(bacco_emulator, cosmo, target_z)
        target_sum += target.mean()
        target_sq_sum += (target ** 2).mean()

        n += 1

    # Compute means and stds
    scalers = {
        "solver_z": solver_z.tolist(),
        "target_z": target_z.tolist(),
        "features": {},
        "target": {}
    }

    # Feature scalers (per-bin for cosmological features like H, D, etc.)
    for key in list(func_map.keys()):
        mean = feature_sums[key] / n
        variance = (feature_sq_sums[key] / n) - mean ** 2
        std = np.sqrt(np.maximum(variance, 1e-8))
        scalers["features"][key] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
        }

    # target scaler (single float mean/std across all bins and redshifts)
    target_mean = target_sum / n 
    target_var = (target_sq_sum / n) - target_mean ** 2
    target_std = np.sqrt(np.maximum(target_var, 1e-8))
    scalers["target"] = {
        "mean": float(target_mean),
        "std": float(target_std),
    }
    return scalers


def main():
    parser = argparse.ArgumentParser(description="Compute and save scalers for all features")
    parser.add_argument("--train_cosmologies_file", type=str, default="data/cosmologies_train.csv")
    parser.add_argument("--output", type=str, default="data/scalers.json")
    parser.add_argument("--use_every_n", type=int, default=1)

    args = parser.parse_args()

    bacco_emulator = Matter_powerspectrum()

    scalers = compute_scalers(
        bacco_emulator, 
        args.train_cosmologies_file, 
        bacco_target_z,
        args.use_every_n
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(scalers, f, indent=2)

    print(f"\nScalers saved to {output_path}")


if __name__ == "__main__":
    main()
