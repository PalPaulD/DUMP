"""
Training script for cosmology emulator using PyTorch Lightning.
Supports YAML configs and CLI overrides.
"""
# I am so tired of baccoemu warnings...
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='keras')

import argparse
import yaml
import json
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from torch.utils.data import DataLoader
from pathlib import Path

from DUMP.models import NeuralODE
from DUMP.data.bacco_Pk import BaccoPk
from DUMP.data.constants import bacco_k, bacco_target_z
from DUMP.plotting import plot_errors_w0wa_dataset, plot_one_param_ratios, plot_errors_redshift_k


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_scalers(scalers_path, features_list):
    """Load scalers from JSON file and filter for needed features."""
    with open(scalers_path, 'r') as f:
        data = json.load(f)

    # Convert to torch tensors and filter only needed features
    scalers = {
        "solver_z": data["solver_z"],
        "target_z": data["target_z"],
    }

    # Load feature scalers (including lin_pk)
    for key in features_list:
        if key not in data["features"]:
            raise ValueError(
                f"Feature '{key}' not found in scalers.\n"
                f"Available: {list(data['features'].keys())}\n"
                f"Run: python scripts/compute_scalers.py --train_file <your_train.csv>"
            )
        scalers[key] = {
            "mean": data["features"][key]["mean"],
            "std": data["features"][key]["std"]
        }

    # Load target scaler (nonlin_pk)
    scalers["target"] = {
        "mean": data["target"]["mean"],
        "std": data["target"]["std"]
    }

    return scalers


def main():
    parser = argparse.ArgumentParser(description="Train cosmology emulator")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to YAML config")
    parser.add_argument("--set", nargs="*", default=[],
                        help="Override config: key=value (e.g., --set lr=0.01 batch_size=64)")

    args = parser.parse_args()

    # Load config from YAML
    config = load_config(args.config)

    # Apply CLI overrides
    for override in args.set:
        key, value = override.split("=", 1)
        config[key] = yaml.safe_load(value)

    # Convert to namespace
    args = argparse.Namespace(**config)

    # Setup output directory
    output_path = Path("./experiments") / args.experiment_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dict = vars(args)
    with open(output_path / "config.yaml", "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    print(f"Config saved to {output_path / 'config.yaml'}")

    # Load scalers
    print(f"Loading scalers from {args.scalers_path}")
    scalers = load_scalers(args.scalers_path, args.features_list)

    # Create datasets (emulator will be initialized lazily in each worker)
    train_dataset = BaccoPk(
        features_list=args.features_list,
        target_z=bacco_target_z,
        scalers=scalers,
        cosmologies_file=args.train_file,
    )
    val_dataset = BaccoPk(
        features_list=args.features_list,
        target_z=bacco_target_z,
        scalers=scalers,
        cosmologies_file=args.val_file,
    )
    test_dataset = BaccoPk(
        features_list=args.features_list,
        target_z=bacco_target_z,
        scalers=scalers,
        cosmologies_file=args.test_file,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("\nDataset sizes:")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    print(f"Features: {args.features_list}")

    # Create model
    mlp_params = {
        "in_dim": len(args.features_list) + len(bacco_k),   # include Pk in the input of RHS too
        "out_dim": len(bacco_k),
        "width": args.width,
        "depth": args.depth,
    }

    # Account shapes for spectra in features
    if "lin_pk" in args.features_list:
        mlp_params["in_dim"] += (len(bacco_k) - 1)

    model = NeuralODE(
        mlp_params=mlp_params,
        features_list=args.features_list,
        lr=args.lr,
        lr_factor=args.lr_factor,
        scheduler_patience=args.scheduler_patience,
        scalers=scalers,
        val_with_desi_corner=args.val_with_desi_corner
    )
    print(f"MLP: {model.mlp}")

    # Setup logger
    if args.use_wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.experiment_name,
            save_dir=str(output_path),
            config=vars(args),  # Log all config parameters to wandb
        )
    else:
        logger = CSVLogger(
            save_dir=str(output_path),
            name=args.experiment_name
        )
        print("Running with no wandb! The results are stored locally")


    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor="val L2 DESI corner" if args.val_with_desi_corner else "val L2 full",
            patience=args.patience,
            mode="min",
            verbose=True,
            min_delta=1e-10
        ),
        ModelCheckpoint(
            filename=f"best_{args.experiment_name}",
            dirpath=output_path / "weights",
            monitor="val L2 DESI corner",
            mode="min",
            save_top_k=1,
            verbose=True
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Create trainer
    trainer = pl.Trainer(
        num_sanity_val_steps=1,
        log_every_n_steps=1,
        max_steps=args.max_train_steps,
        val_check_interval=args.val_check_interval,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=str(output_path),
        enable_progress_bar=True,
    )

    # Train
    print(f"Starting training: {args.experiment_name}")

    trainer.fit(model, train_loader, val_loader)

    print(f"\nTraining complete!")

    print("Running evaluation plotting...")
    plot_dir = output_path / "plots"
    plot_dir.mkdir(exist_ok=True)

    # w0wa plot requires a lot of samples, so I do it with dataloaders
    print("Making w0wa errors plots...")
    test_pred = trainer.predict(model, test_loader)
    w0 = np.concatenate([batch['cosmo']['w0'].cpu().numpy() for batch in test_pred])
    wa = np.concatenate([batch['cosmo']['wa'].cpu().numpy() for batch in test_pred])
    target = 10 ** np.concatenate([batch['target'].cpu().numpy() for batch in test_pred])
    pred = 10 ** np.concatenate([batch['target_pred'].cpu().numpy() for batch in test_pred])
    plot_errors_w0wa_dataset(
        w0=w0,
        wa=wa,
        target=target,
        target_pred=pred,
        target_z=model.target_z.cpu().numpy(),
        k=bacco_k,
        kmin=args.plot_kmin,
        kmax=args.plot_kmax,
        logger=logger if args.use_wandb else None,
        save_location=str(plot_dir) if not args.use_wandb else "wandb",
    )

    print("Making ratio plots...")
    plot_one_param_ratios(
        trained_model=model,
        logger=logger if args.use_wandb else None,
        samples_per_param=7,
        save_location=str(plot_dir) if not args.use_wandb else "wandb",
    )

    print("Making k-z interpolation error heatmap...")
    plot_errors_redshift_k(
        trained_model=model,
        logger=logger if args.use_wandb else None,
        save_location=str(plot_dir) if not args.use_wandb else "wandb",
    )

    print(f"\nPlots saved to {plot_dir}")
    print(f"\nExperiment complete: {args.experiment_name}")
    print(f"Output directory: {output_path}")


if __name__ == "__main__":
    main()
