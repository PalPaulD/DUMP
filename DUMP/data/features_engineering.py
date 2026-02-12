import numpy as np
from DUMP.data.constants import bacco_k

import torch



###
#       Spectra (linear and nonlinear)
###
def nonlin_pk(
    emulator,
    cosmo_params: dict[np.float64], 
    z: np.ndarray, 
    k: np.ndarray = bacco_k,
):
    k_, Pk = emulator.get_nonlinear_pk(
        k=k,
        cold=True,
        expfactor=1/(1+z),
        omega_cold=cosmo_params["omega_cold"],
        omega_baryon=cosmo_params["omega_baryon"],
        hubble=cosmo_params["hubble"],
        w0=cosmo_params["w0"],
        wa=cosmo_params["wa"],
        sigma8_cold=cosmo_params["sigma8_cold"],
        ns=cosmo_params["ns"],
        neutrino_mass=0.0
    )
    assert np.allclose(k_, k), "Emulator returned k values that do not match input k values"
    return np.log10(Pk)

def lin_pk(
    emulator,
    cosmo_params: dict[np.float64], 
    z: np.ndarray,
    k: np.ndarray = bacco_k
):
    k_, Pk = emulator.get_linear_pk(
        k=k,
        cold=True,
        expfactor=1/(1+z),
        omega_cold=cosmo_params["omega_cold"],
        omega_baryon=cosmo_params["omega_baryon"],
        hubble=cosmo_params["hubble"],
        w0=cosmo_params["w0"],
        wa=cosmo_params["wa"],
        sigma8_cold=cosmo_params["sigma8_cold"],
        ns=cosmo_params["ns"],
        neutrino_mass=0.0
    )
    assert np.allclose(k_, k), "Emulator returned k values that do not match input k values"
    return np.log10(Pk)

###
#    Cosmological background quantities
###

def I(
    cosmo_params: dict[np.float64],
    z: np.ndarray
):
    '''
    Compute I(z) = int_0^z 1 + w(z') / (1 + z') dz' where w(z) = w0 + wa * z / (1 + z).
    Utility function to be used by other features functions.
    '''
    return (1 + cosmo_params["w0"] + cosmo_params["wa"]) * np.log(1 + z) - cosmo_params["wa"] * (z / (1 + z))

def dI_dz(
    cosmo_params: dict[np.float64],
    z: np.ndarray
):
    '''
    Compute dI/dz = [1 + w(z)] / (1 + z) where w(z) = w0 + wa * z / (1 + z)
    '''
    return (1 + cosmo_params["w0"] + cosmo_params["wa"]) / (1 + z) - cosmo_params["wa"] / (1 + z)**2

def H(
    cosmo_params: dict[np.float64],
    z: np.ndarray,
):
    '''
    Compute H(z) = H0 * sqrt( omega_m * (1 + z)^3 + omega_lambda * exp(3 * I) )
    where I = int_0^z 1 + w(z') / (1 + z') dz' computed by function I()
    Utility function to be used by other features functions.
    '''
    H0 = 100 * cosmo_params["hubble"]
    omega_cold = cosmo_params["omega_cold"]
    I_ = I(cosmo_params, z)

    H = H0 * np.sqrt( omega_cold * (1 + z)**3 + (1 - omega_cold) * np.exp(3 * I_) )
    return H

def dH_dz(
    cosmo_params: dict[np.float64],
    z: np.ndarray,
):
    '''
    Compute dH/dz = H0^2
    '''
    I_ = I(cosmo_params, z)
    dI_dz_ = dI_dz(cosmo_params, z)
    H_ = H(cosmo_params, z)
    H0 = 100 * cosmo_params["hubble"]
    omega_cold = cosmo_params["omega_cold"]

    dH_dz = (H0**2 / (2 * H_)) * ( 3 * omega_cold * (1 + z)**2 + 3 * (1 - omega_cold) * np.exp(3 * I_) * dI_dz_ )
    return dH_dz

def logrhom(cosmo_params: dict[np.float64], z: np.ndarray):
    '''
    Compute log(rho_m(z)) = log(omega_m * rho_crit * (1 + z)^3)
    '''
    H0 = 100 * cosmo_params["hubble"]
    omega_cold = cosmo_params["omega_cold"]
    logrhom = np.log(H0**2 * omega_cold * (1 + z)**3)
    return logrhom

def dlogrhom_dz(cosmo_params: dict[np.float64], z: np.ndarray):
    '''
    Compute dlog(rho_m)/dz = 3 / (1 + z)
    '''
    return 3 / (1 + z)

def D(
    cosmo_params: dict[np.float64],
    z: np.ndarray,
    z_inf: float = 500.0,
    n_grid: int = 512
):
    Om = cosmo_params["omega_cold"]
    w0 = cosmo_params["w0"]
    wa = cosmo_params["wa"]
    Ode = 1 - Om

    t = np.logspace(-8, 0, n_grid)
    z_grid = z[:, None] + t * (z_inf - z[:, None])

    de = (1 + z_grid)**(3 * (1 + w0 + wa)) * np.exp(-3 * wa * z_grid / (1 + z_grid))
    E_grid = np.sqrt(Om * (1 + z_grid)**3 + Ode * de)
    integrand = (1 + z_grid) / E_grid**3

    integral = np.trapezoid(integrand, z_grid, axis=1)
    D_unnorm = E_grid[:, 0] * integral

    z0_grid = t * z_inf
    de0 = (1 + z0_grid)**(3 * (1 + w0 + wa)) * np.exp(-3 * wa * z0_grid / (1 + z0_grid))
    E0_grid = np.sqrt(Om * (1 + z0_grid)**3 + Ode * de0)
    integrand0 = (1 + z0_grid) / E0_grid**3
    D_at_0 = E0_grid[0] * np.trapezoid(integrand0, z0_grid)

    return D_unnorm / D_at_0


def dD_dz(
    cosmo_params: dict[np.float64],
    z: np.ndarray,
    z_inf: float = 500.0,
    n_grid: int = 512
):
    D_ = D(cosmo_params, z)
    Om = cosmo_params["omega_cold"]
    w0 = cosmo_params["w0"]
    wa = cosmo_params["wa"]
    Ode = 1 - Om

    de_z = (1 + z)**(3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))
    E_z = np.sqrt(Om * (1 + z)**3 + Ode * de_z)

    d_rho_de_dz = de_z * (3 * (1 + w0 + wa) / (1 + z) - 3 * wa / (1 + z)**2)
    dE_dz = (3 * Om * (1 + z)**2 + Ode * d_rho_de_dz) / (2 * E_z)

    t = np.logspace(-8, 0, n_grid)
    z0_grid = t * z_inf
    de0 = (1 + z0_grid)**(3 * (1 + w0 + wa)) * np.exp(-3 * wa * z0_grid / (1 + z0_grid))
    E0_grid = np.sqrt(Om * (1 + z0_grid)**3 + Ode * de0)
    integrand0 = (1 + z0_grid) / E0_grid**3
    D_at_0 = E0_grid[0] * np.trapezoid(integrand0, z0_grid)

    # Assumes that D(z=0) = 1
    return (dE_dz / E_z) * D_- (1 + z) / (E_z**2)

def z(
    cosmo_params: dict[np.float64],
    z: np.ndarray,
):
    return z

###
#   Simple cosmological parameters as features
###

def ns(
    cosmo_params: dict[np.float64],
    z: np.ndarray
):
    # Notice that these functions imply batching over z
    ns = cosmo_params["ns"] * np.ones_like(z)
    return ns

def sigma8(
    cosmo_params: dict[np.float64],
    z: np.ndarray
):
    sigma8 = cosmo_params["sigma8_cold"] * np.ones_like(z)
    return sigma8


###
#    Main feature engineering function
###
# IF NEW FEATURES ARE CREATED, THEIR FUNCTION SHOULD BE REGISTERED HERE!
func_map = {
    "z": z,
    "H": H,
    "dH_dz": dH_dz,
    "logrhom": logrhom,
    "dlogrhom_dz": dlogrhom_dz,
    "D": D,
    "dD_dz": dD_dz,
    "lin_pk": lin_pk,
    "sigma8": sigma8,
    "ns": ns
}

def make_features(
    bacco_emulator,
    features_list: list[str],
    cosmo_params: dict[np.float64],
    z: np.ndarray
):
    '''
    This function orchestrates the computation of all the features we want to use for our ML model, given a set of cosmological parameters and a redshift.
    It is meant to be used for a single cosmology and an array of redshifts, so a torch dataset can call it for each cosmology (batch element) separately
    but get results for all requested redshifts.

    Only computes the features that are specified in features_keys, which is a list of strings. This allows us to easily add or remove features without having to change the code in multiple places.
    '''


    # Assemble features, spectra require bacco emulator
    features = {}
    for key in features_list:
        if key not in func_map:
            raise ValueError(f"Feature {key} is not implemented. Available features are: {list(func_map.keys())}")
        else:
            if key=="lin_pk" or key=="nonlin_pk":
                features[key] = func_map[key](bacco_emulator, cosmo_params, z)
            else:
                features[key] = func_map[key](cosmo_params, z)
    return features


def flatten_and_scale_features(
    features_dict: dict[np.ndarray], 
    scalers: dict, 
    features_list: list[str]
):
    """Convert features dict to scaled numpy array."""
    features_out = []
    for key in features_list:
        feature = features_dict[key]
        feat_scaled = (feature - scalers[key]["mean"]) / scalers[key]["std"]

        if feat_scaled.ndim == 1:
            feat_scaled = np.expand_dims(feat_scaled, -1)

        features_out.append(feat_scaled)

    return np.concatenate(features_out, axis=-1)