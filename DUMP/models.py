import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchdiffeq import odeint_adjoint as odeint
from DUMP.data.constants import bacco_target_z, solver_dz
from DUMP.utils import find_solver_grid 
from DUMP.data.features_engineering import make_features, nonlin_pk

class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        width,
        depth,
        activation=nn.ReLU,
    ):
        super().__init__()

        layers = []
        for i in range(depth):
            layers += [nn.Linear(in_dim, width), activation()]
            in_dim = width

        layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class NeuralODE(pl.LightningModule):
    def __init__(
        self,
        mlp_params,
        features_list,
        lr,
        lr_factor,
        scheduler_patience,
        scalers,
        val_with_desi_corner=True
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["scalers"])

        self.features_list = features_list

        # Validate scalers z-grids match model z-grids
        solver_z = find_solver_grid(bacco_target_z, solver_dz)
        if "solver_z" in scalers and "target_z" in scalers:
            saved_solver_z = np.array(scalers["solver_z"])
            saved_target_z = np.array(scalers["target_z"])

            if not np.allclose(saved_solver_z, solver_z, rtol=1e-5):
                raise ValueError(
                    f"Solver z mismatch between scalers and model!\n"
                    f"  Model: [{solver_z[0]:.2f}, {solver_z[-1]:.2f}] ({len(solver_z)} points)\n"
                    f"  Scalers: [{saved_solver_z[0]:.2f}, {saved_solver_z[-1]:.2f}] ({len(saved_solver_z)} points)"
                )

            if not np.allclose(saved_target_z, bacco_target_z, rtol=1e-5):
                raise ValueError(
                    f"Target z mismatch between scalers and model!\n"
                    f"  Model: {bacco_target_z}\n"
                    f"  Scalers: {saved_target_z}"
                )

        # Prep grids
        self.set_solver_grid(solver_z, bacco_target_z)

        # Validate features have scalers
        for f in features_list:
            if f not in scalers:
                raise ValueError(f"Feature '{f}' not in scalers.")

        # Setup MLP
        self.mlp = MLP(**mlp_params)

        # Learning params
        self.lr = lr
        self.lr_factor = lr_factor
        self.scheduler_patience = scheduler_patience
        self.val_with_desi_corner = val_with_desi_corner

        # Store all scalers as buffers (saved with checkpoint)
        self.register_buffer("target_mean", torch.as_tensor(scalers["target"]["mean"], dtype=torch.float32))
        self.register_buffer("target_std", torch.as_tensor(scalers["target"]["std"], dtype=torch.float32))

        for f in features_list:
            self.register_buffer(f"{f}_mean", torch.as_tensor(scalers[f]["mean"], dtype=torch.float32))
            self.register_buffer(f"{f}_std", torch.as_tensor(scalers[f]["std"], dtype=torch.float32))

    def set_solver_grid(self, solver_z, target_z):

        # Store grids
        self.register_buffer("solver_z", torch.tensor(solver_z.copy(), dtype=torch.float32))
        self.register_buffer("target_z", torch.tensor(target_z.copy(), dtype=torch.float32))

        # Validate uniform spacing (required for RK4)
        spacings = self.solver_z[1:] - self.solver_z[:-1]
        if not torch.allclose(spacings, spacings[0], rtol=1e-4, atol=1e-7):
            raise ValueError(
                f"solver_z must be uniformly spaced. "
                f"Spacing varies: {spacings.min():.10f} to {spacings.max():.10f}\n"
            )

        # Validate target_z is a subset of solver_z
        for tz in target_z:
            idx = torch.argmin(torch.abs(self.solver_z - tz))
            closest = self.solver_z[idx]
            if torch.abs(closest - tz) > 1e-6:
                raise ValueError(
                    f"Target z={tz:.6f} not in solver grid! "
                    f"Closest: {closest:.6f}"
                )


    def forward(self, features, init_cond):
        """Training: solve on feature grid, extract at target grid."""

        def rhs(z, y):
            idx = torch.argmin(torch.abs(self.solver_z - z))
            # Is the solver's redshift on the grid?
            closest_z = self.solver_z[idx]
            if torch.abs(closest_z - z) > 1e-6:
                raise ValueError(f"ODE solver at z={z:.6f}, not on grid (closest: {closest_z:.6f})")
            # Cat features and current stat, do forward pass
            features_z = features[:, idx]
            x = torch.cat([y, features_z], dim=-1)
            return self.mlp(x)

        solution = odeint(
            rhs,
            init_cond,
            self.target_z,
            method="rk4",
            adjoint_params=tuple(self.mlp.parameters()),
            options={"step_size": solver_dz}
        )

        # solution shape: (len(target_z), batch, features) -> swap to (batch, len(target_z), features)
        # Skip the initial condition (first time step) by taking [1:]
        solution = solution.swapaxes(0, 1)
        return solution[:, 1:, :]

    def training_step(self, batch, batch_idx):
        _, features, init_cond, target = batch
        target_pred = self(features, init_cond)
        loss = F.mse_loss(target_pred, target)  # Train to fit redshifts together
        self.log("train L2", loss, on_step=True, on_epoch=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        cosmo, features, init_cond, target = batch
        target_pred = self(features, init_cond)
        loss = F.mse_loss(target_pred, target, reduction="none").mean(dim=-1)   # average over bins, not over samples

        # Loss over the whole range of params (with batch size for proper weighted averaging)
        self.log("val L2 full", loss.mean(), prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(loss))

        # Loss over the DESI corner of params (only log if batch has DESI corner samples)
        mask = (cosmo["w0"] > -1.0) & (cosmo["wa"] < 0.0)
        if mask.any():
            loss_desi = loss[mask].mean()
            # Only log when we have samples - batches without samples won't contribute
            # Specify batch_size so Lightning knows how many samples contributed (for weighted average)
            self.log("val L2 DESI corner", loss_desi, prog_bar=True, on_step=False, on_epoch=True,
                     batch_size=mask.sum().item())

    def predict_step(self, batch, batch_idx):
        '''
        This is meant to be used with dataloaders
        '''
        cosmo, features, init_cond, target = batch
        pred = self(features, init_cond)

        # Unscale
        target_unscaled = target * self.target_std + self.target_mean
        pred_unscaled = pred * self.target_std + self.target_mean
        return {
            "cosmo": cosmo,
            "target": target_unscaled,
            "target_pred": pred_unscaled
        }

    @torch.no_grad()
    def inference(self, cosmo, bacco_emulator):
        """Inference: compute features, solve, return unscaled Pk."""
        # Compute raw features and initial condition
        features_dict = make_features(bacco_emulator, self.features_list, cosmo, self.solver_z.cpu().numpy())
        init_cond_raw = nonlin_pk(bacco_emulator, cosmo, self.target_z)[0]

        # Convert to torch and scale features
        features_list = []
        for f in self.features_list:
            f_tensor = torch.tensor(features_dict[f], dtype=torch.float32)
            f_scaled = (f_tensor - getattr(self, f"{f}_mean")) / getattr(self, f"{f}_std")
            features_list.append(f_scaled.unsqueeze(-1) if f_scaled.ndim == 1 else f_scaled)   # looking at you, lin_pk
        features = torch.cat(features_list, dim=-1).unsqueeze(0)

        # Scale initial condition
        init_cond_raw = torch.tensor(init_cond_raw, dtype=torch.float32)
        init_cond = (init_cond_raw - self.target_mean) / self.target_std

        # Solve (reuse forward method)
        solution = self(features, init_cond.unsqueeze(0))

        # Unscale and return
        return (solution * self.target_std + self.target_mean).squeeze(0)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=self.lr_factor, patience=self.scheduler_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train L2",  # Metric to watch
                "interval": "step",  # 'epoch' or 'step'
                "frequency": 1,
                "strict": True,
            },
        }