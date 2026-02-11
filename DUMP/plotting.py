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

# Configure matplotlib for publication-quality physics plots
plt.rcParams.update({
    'font.family': 'serif',
    'text.usetex': False,  # Set to True if LaTeX is available
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
})

def plot_errors_w0wa_dataset(
    w0: np.ndarray,
    wa: np.ndarray,
    target: np.ndarray,
    target_pred: np.ndarray,
    target_z: np.ndarray,
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
    rel_errors = np.mean(np.abs(rel_errors), axis=-1)

    # Make plots for prediction bins (target_z[0] is init cond)
    for i, z_val in enumerate(target_z[1:]):
        for res in resolutions:
            stat, x_edges, y_edges, _ = binned_statistic_2d(w0, wa, rel_errors[:,i], statistic="mean", bins=res)
            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.imshow(
                stat.T,
                origin="lower",
                extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                cmap="viridis",
                aspect="auto",
            )
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Relative error (%)", fontsize=10)
            ax.set_xlabel(r"$w_0$", fontsize=11)
            ax.set_ylabel(r"$w_a$", fontsize=11)
            ax.set_title(f"Mean error at $z={z_val:.1f}$ for $k \\in [{kmin:.1f}, {kmax:.1f}]$ $h$/Mpc", fontsize=11)
            fig.tight_layout()

            # Be optional about using wandb
            if save_location=="wandb":
                logger.experiment.log({f"w0wa_errors/{res}res/z{z_val:.1f}": wandb.Image(fig)})
            else:
                save_path = Path(save_location) / "w0wa_errors" / f"{res}res"
                save_path.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path / f"z{z_val:.1f}.png", dpi=200, bbox_inches='tight')
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

    # Initialize bacco emulator (suppress warnings)
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        bacco_emulator = baccoemu.Matter_powerspectrum()

    # Use model's target_z grid for consistency
    target_z = trained_model.target_z.cpu().numpy()

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
    k_, fiducial_bacco = bacco_emulator.get_nonlinear_pk(
        k=bacco_k,
        cold=True,
        expfactor=1/(1+target_z[1:]),   # drop the first redshift, it is init cond
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

            # Get predictions using model's target_z
            bacco_ = nonlin_pk(bacco_emulator, cosmo, target_z)[1:]
            pred_ = trained_model.inference(cosmo, bacco_emulator).cpu().numpy()
            bacco.append(bacco_)
            preds.append(pred_)

        # shape (samples_per_param, n_z, n_k)
        # Convert from log10 to linear scale for ratios
        bacco = 10 ** np.array(bacco)
        preds = 10 ** np.array(preds)
        bacco_ratios = bacco / fiducial_bacco
        model_ratios = preds / fiducial_pred

        # Plot for each redshift bin
        for z_idx in range(bacco_ratios.shape[1]):
            fig, ax = plt.subplots(figsize=(8, 5))

            # Use a professional color palette
            colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, samples_per_param))

            # Plot each parameter variation
            for i, param_val in enumerate(param_values):
                ax.plot(
                    bacco_k, bacco_ratios[i, z_idx, :],
                    color=colors[i], linestyle='-', linewidth=1.5, alpha=0.7,
                    label=f'{param_label}={param_val:.3f}'
                )
                ax.plot(
                    bacco_k, model_ratios[i, z_idx, :],
                    color=colors[i], linestyle='--', linewidth=1.5, alpha=0.9
                )

            # Add reference line
            ax.axhline(1.0, color='black', linestyle=':', linewidth=1.0, alpha=0.6, zorder=0)

            ax.set_xlabel(r'$k$ [$h$ Mpc$^{-1}$]', fontsize=11)
            ax.set_ylabel(r'$P(k) / P_{\rm fid}(k)$', fontsize=11)
            ax.set_title(f'{param_label} variation, $z={target_z[z_idx]:.2f}$', fontsize=11)
            ax.set_xscale('log')
            ax.legend(fontsize=7, ncol=1, frameon=True, fancybox=False, edgecolor='black', loc='best')
            ax.grid(True, alpha=0.2, linewidth=0.5)
            fig.tight_layout()

            # Save
            if save_location == "wandb":
                logger.experiment.log({
                    f"param_ratios/z{target_z[z_idx]:.1f}/{param_name}": wandb.Image(fig)
                })
            else:
                save_path = Path(save_location) / "param_ratios" / f"z{target_z[z_idx]:.1f}"
                save_path.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path / f"{param_name}.png", dpi=200, bbox_inches='tight')

            plt.close(fig)

def plot_errors_redshift_k(
    trained_model: NeuralODE,
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
        expfactor=1/(1+z[1:]),      # cut the initial redshift, not predicted by emu 
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
    fig, ax = plt.subplots(figsize=(9, 5))

    # Create meshgrid for pcolormesh (use z[1:] to match prediction dimensions)
    K, Z = np.meshgrid(k, z[1:])

    # Plot heatmap
    im = ax.pcolormesh(
        K, Z, rel_errors,
        cmap='YlOrRd',
        shading='auto',
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Relative error (%)', fontsize=10)

    # Formatting
    ax.set_xlabel(r'$k$ [$h$ Mpc$^{-1}$]', fontsize=11)
    ax.set_ylabel(r'Redshift $z$', fontsize=11)
    ax.set_title('Model accuracy at fiducial cosmology', fontsize=11)
    ax.set_xscale('log')

    fig.tight_layout()

    # Save
    if save_location == "wandb":
        logger.experiment.log({"errors_k_z": wandb.Image(fig)})
    else:
        save_path = Path(save_location) / "performance"
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / "errors_k_z.png", dpi=200, bbox_inches='tight')

    plt.close(fig)
