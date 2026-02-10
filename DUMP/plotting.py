import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from pathlib import Path
import wandb
import baccoemu
import os

from DUMP.data.constants import bacco_k
from DUMP.models import NeuralODE

import contextlib

def plot_errors_w0wa_dataset(
    w0: np.ndarray,
    wa: np.ndarray,
    target: np.ndarray,
    target_pred: np.ndarray,
    k: np.ndarray = bacco_k,
    kmin: float = 0.5,      # in units of Mpc/h
    kmax: float = 3.0,
    logger=None,
    save_location: str = "wandb",
    resolutions: list[int] = [50, 100, 150]
): 
    # Filter out large scales and "too small" scales
    mask = np.logical_and(k > kmin, k < kmax)
    rel_errors = 100 * (target_pred[:,:,mask] - target[:,:,mask]) / target[:,:,mask]
    rel_errors = np.mean(np.abs(rel_errors), axis=1)

    # Make plots for all bins
    for i in range(len(rel_errors.shape[1])):
        for res in resolutions:
            stat, x_edges, y_edges, _ = binned_statistic_2d(w0, wa, rel_errors[:,i], statistic="mean", bins=res)
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(
                stat.T,
                origin="lower",
                extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                cmap="viridis",
                aspect="auto",
            )
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("%")
            ax.set(
                xlabel="w0", ylabel="wa",
                title=f"Mean rel error in spectrum for k in [{kmin}, {kmax}]h/Mpc\nbinning res: {res}x{res}",
            )

            # Be optional about using wandb
            if save_location=="wandb":
                logger.experiment.log({f"w0wa_errors/performance_{res}res/{i}_z_bin": wandb.Image(fig)})
            else:
                plt.savefig(Path(save_location) / "w0wa_errors" / f"{res}res" / f"{i}_z_bin.png")
            plt.close(fig)

def plot_one_param_ratios(
    trained_model: NeuralODE,
    logger=None,
    samples_per_param: int = 7,
    save_location: str = "wandb",
    Om0_range: tuple = (0.23, 0.4),
    Ob0_range: tuple = (0.04, 0.06),
    sigma8_range: tuple = (0.73, 0.9),
    ns_range: tuple = (0.92, 1.01),
    hubble_range: tuple = (0.6, 0.8),
    w0_range: tuple = (-1.15, -0.85),
    wa_range: tuple = (-0.3, 0.3),
    fiducial_cosmo={
        "omega_cold": 0.3175, "omega_baryon": 0.049, "hubble": 0.6711,
        "w0": -1.0, "wa": 0.0, "sigma8_cold": 0.834, "ns": 0.9624,
    },
):
    """
    Plot P(k) ratios for one-parameter variations.
    For each parameter, creates plots showing prediction/fiducial for both
    bacco emulator and trained model across different redshift bins.
    """
    from DUMP.data.features_engineering import nonlin_pk
    from DUMP.data.constants import bacco_target_z

    # Initialize bacco emulator (suppress warnings)
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        bacco_emulator = baccoemu.Matter_powerspectrum()

    # Define parameters to vary
    params_info = {
        'omega_cold': (r'$\Omega_m$', Om0_range),
        'omega_baryon': (r'$\Omega_b$', Ob0_range),
        'sigma8_cold': (r'$\sigma_8$', sigma8_range),
        'ns': (r'$n_s$', ns_range),
        'hubble': (r'$h$', hubble_range),
        'w0': (r'$w_0$', w0_range),
        'wa': (r'$w_a$', wa_range),
    }

    # Get fiducial predictions (both return log10(P(k)) with shape (n_z, n_k))
    #fiducial_bacco = nonlin_pk(bacco_emulator, fiducial_cosmo, bacco_target_z)
    k_, fiducial_bacco = bacco_emulator.get_nonlinear_pk(
        k=bacco_k,
        cold=True,
        expfactor=1/(1+trained_model.target_z),
        omega_cold=fiducial_cosmo["omega_cold"],
        omega_baryon=fiducial_cosmo["omega_baryon"],
        hubble=fiducial_cosmo["hubble"],
        w0=fiducial_cosmo["w0"],
        wa=fiducial_cosmo["wa"],
        sigma8_cold=fiducial_cosmo["sigma8_cold"],
        ns=fiducial_cosmo["ns"],
        neutrino_mass=0.0
    )
    assert np.allclose(k_, bacco_k)
    fiducial_pred = 10 ** trained_model.inference(fiducial_cosmo, bacco_emulator).cpu().numpy()

    # For each parameter
    for param_name, (param_label, param_range) in params_info.items():
        # Create parameter variations
        param_values = np.linspace(param_range[0], param_range[1], samples_per_param)

        bacco = []
        preds = []
        for param_val in param_values:
            # Create cosmology with varied parameter
            cosmo = fiducial_cosmo.copy()
            cosmo[param_name] = param_val

            # Get predictions
            bacco_ = nonlin_pk(bacco_emulator, cosmo, bacco_target_z)
            pred_ = trained_model.inference(cosmo, bacco_emulator).cpu().numpy()
            bacco.append(bacco_)
            preds.append(pred_)

        # shape (samples_per_param, n_z, n_k)
        bacco = np.array(bacco)
        preds = np.array(preds)
        bacco_ratios = bacco / fiducial_bacco
        model_ratios = preds / fiducial_pred

        # Plot for each redshift bin
        for z_idx in range(bacco_ratios.shape[1]):
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot each parameter variation
            for i, param_val in enumerate(param_values):
                # Use different colors for bacco vs model
                color = plt.cm.viridis(i / (samples_per_param - 1))
                ax.plot(
                    bacco_k, bacco_ratios[i, z_idx, :],
                    color=color, linestyle='-', linewidth=2,
                    label=f'Bacco {param_label}={param_val:.3f}'
                )
                ax.plot(
                    bacco_k, model_ratios[i, z_idx, :],
                    color=color, linestyle='--', linewidth=2,
                    label=f'Model {param_label}={param_val:.3f}'
                )

            ax.axhline(1.0, color='black', linestyle=':', alpha=0.5)
            ax.set(
                xlabel='k [h/Mpc]',
                ylabel=r'$P(k) / P_{\rm fiducial}(k)$',
                title=f'{param_label} variation, z={bacco_target_z[z_idx]:.2f}',
                xscale='log',
            )
            ax.legend(fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)

            # Save
            if save_location == "wandb":
                logger.experiment.log({
                    f"param_ratios/{z_idx}_z_bin/{param_name}": wandb.Image(fig)
                })
            else:
                save_path = Path(save_location) / "param_ratios" / f"{z_idx}_z_bin"
                save_path.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path / f"paran_name.png")

            plt.close(fig)

def plot_errors_redshift_k(
    trained_model: NeuralODE,
    new_solver_z: np.ndarray,
    logger=None,
    save_location: str = "wandb",
    fiducial_cosmo={
        "omega_cold": 0.3175, "omega_baryon": 0.049, "hubble": 0.6711,
        "w0": -1.0, "wa": 0.0, "sigma8_cold": 0.834, "ns": 0.9624,
    },
):
    """
    Plot relative errors as a 2D heatmap of k vs redshift for fiducial cosmology.
    Tests model interpolation performance across the (k, z) space.

    Uses the model's own target_z grid, allowing assessment of models
    trained with different redshift grids.

    Parameters:
    -----------
    trained_model : NeuralODE
        Trained model (uses its own target_z grid)
    logger : optional
        WandB logger
    save_location : str
        "wandb" or path to save directory
    fiducial_cosmo : dict
        Fiducial cosmology to evaluate
    """
    from DUMP.data.features_engineering import nonlin_pk

    # Initialize bacco emulator (suppress warnings)
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        bacco_emulator = baccoemu.Matter_powerspectrum()

    # Get model's target redshift grid (may differ from default)
    z = trained_model.target_z.cpu().numpy()
    k = bacco_k

    # Get predictions at fiducial cosmology (both return log10(P(k)))
    k_, bacco_pred = bacco_emulator.get_nonlinear_pk(
        k=k,
        cold=True,
        expfactor=1/(1+z),
        omega_cold=fiducial_cosmo["omega_cold"],
        omega_baryon=fiducial_cosmo["omega_baryon"],
        hubble=fiducial_cosmo["hubble"],
        w0=fiducial_cosmo["w0"],
        wa=fiducial_cosmo["wa"],
        sigma8_cold=fiducial_cosmo["sigma8_cold"],
        ns=fiducial_cosmo["ns"],
        neutrino_mass=0.0
    )
    assert np.allclose(k_, k)

    model_pred = 10 ** trained_model.inference(fiducial_cosmo, bacco_emulator).cpu().numpy()

    # Compute relative errors: shape (n_z, n_k)
    rel_errors = 100 * np.abs((model_pred - bacco_pred) / bacco_pred)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create meshgrid for pcolormesh
    K, Z = np.meshgrid(k, z)

    # Plot heatmap
    im = ax.pcolormesh(
        K, Z, rel_errors,
        cmap='viridis',
        shading='auto',
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Mean relative error (%)', fontsize=12)

    # Formatting
    ax.set(
        xlabel='k [h/Mpc]',
        ylabel='Redshift z',
        title='Model Interpolation Performance at Fiducial Cosmology',
        xscale='log',
    )
    ax.invert_yaxis()  # Higher redshift at top

    # Add contour lines for readability
    contours = ax.contour(
        K, Z, rel_errors,
        colors='white',
        alpha=0.3,
        linewidths=0.5,
        levels=5
    )
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f%%')

    # Save
    if save_location == "wandb":
        logger.experiment.log({"performance/errors_k_z_heatmap": wandb.Image(fig)})
    else:
        save_path = Path(save_location) / "performance"
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / "errors_k_z_heatmap.png", dpi=150, bbox_inches='tight')

    plt.close(fig)
