"""
Minimal test suite comparing feature engineering to Colossus reference implementation.
Tests a single cosmology across many redshifts.
"""
import numpy as np
import pytest
from colossus.cosmology import cosmology
from DUMP.data.features_engineering import H, dH_dz, logrhom, dlogrhom_dz, D, dD_dz


# Test cosmology parameters
TEST_COSMO = {
    "omega_cold": 0.27,
    "omega_baryon": 0.045,
    "hubble": 0.70,
    "w0": -0.9,
    "wa": 0.1,
    "sigma8_cold": 0.8,
    "ns": 0.96
}

# Test redshifts
Z_TEST = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])

# Relative tolerance for comparisons
RTOL = 2e-3


def setup_colossus_cosmology():
    """Initialize Colossus with our test cosmology."""
    params = {
        'flat': True,
        'H0': TEST_COSMO['hubble'] * 100,
        'Om0': TEST_COSMO['omega_cold'],
        'Ob0': TEST_COSMO['omega_baryon'],
        'sigma8': TEST_COSMO['sigma8_cold'],
        'ns': TEST_COSMO['ns'],
        'de_model': 'w0wa',
        'w0': TEST_COSMO['w0'],
        'wa': TEST_COSMO['wa']
    }
    cosmology.addCosmology('test_cosmo', params)
    return cosmology.setCosmology('test_cosmo')

def five_point_derivative(func, z, h=1e-5):
    """
    Compute derivative using 5-point stencil.
    f'(z) ≈ [-f(z+2h) + 8f(z+h) - 8f(z-h) + f(z-2h)] / (12h)
    """
    # Handle edge cases near z=0
    z_safe = np.maximum(z, 2*h)

    f_plus2 = func(z_safe + 2*h)
    f_plus1 = func(z_safe + h)
    f_minus1 = func(z_safe - h)
    f_minus2 = func(z_safe - 2*h)

    return (-f_plus2 + 8*f_plus1 - 8*f_minus1 + f_minus2) / (12*h)


def test_hubble_parameter():
    """Test H(z) against Colossus."""
    cosmo = setup_colossus_cosmology()

    # Our implementation (returns in km/s/Mpc)
    H_ours = H(TEST_COSMO, Z_TEST)

    # Colossus (returns in km/s/Mpc)
    H_colossus = np.array([cosmo.Hz(z) for z in Z_TEST])

    np.testing.assert_allclose(H_ours, H_colossus, rtol=RTOL,
                               err_msg="H(z) does not match Colossus")


def test_hubble_derivative():
    """Test dH/dz against numerical derivative of Colossus H(z)."""
    cosmo = setup_colossus_cosmology()

    # Our implementation
    dH_ours = dH_dz(TEST_COSMO, Z_TEST)

    # Numerical derivative of Colossus
    def H_colossus_func(z):
        return cosmo.Hz(z)

    dH_numerical = five_point_derivative(H_colossus_func, Z_TEST)

    np.testing.assert_allclose(dH_ours, dH_numerical, rtol=RTOL,
                               err_msg="dH/dz does not match numerical derivative")


def test_log_matter_density():
    """Test log(ρ_m) against Colossus."""
    cosmo = setup_colossus_cosmology()

    # Our implementation
    logrhom_ours = logrhom(TEST_COSMO, Z_TEST)

    # Colossus rho_m in physical units (Msun/Mpc^3)
    # Then convert to match our units
    rho_m_colossus = np.array([cosmo.rho_m(z) for z in Z_TEST])
    logrhom_colossus = np.log(rho_m_colossus)

    # Note: absolute values may differ due to unit conventions,
    # but relative evolution should match
    # Check that differences are consistent (same offset)
    offset = logrhom_ours - logrhom_colossus
    np.testing.assert_allclose(np.std(offset), 0.0, atol=1e-6,
                               err_msg="log(ρ_m) evolution does not match Colossus")


def test_log_matter_density_derivative():
    """Test dlog(ρ_m)/dz = 3/(1+z) analytically."""
    # This is an exact analytical result, independent of cosmology
    dlogrhom_ours = dlogrhom_dz(TEST_COSMO, Z_TEST)
    dlogrhom_analytical = 3.0 / (1.0 + Z_TEST)

    np.testing.assert_allclose(dlogrhom_ours, dlogrhom_analytical, rtol=1e-10,
                               err_msg="dlog(ρ_m)/dz != 3/(1+z)")


def test_growth_factor():
    """Test D(z) against Colossus."""
    cosmo = setup_colossus_cosmology()

    # Our implementation (normalized to D(0)=1)
    D_ours = D(TEST_COSMO, Z_TEST)

    # Colossus growth factor (already normalized to D(0)=1)
    D_colossus = np.array([cosmo.growthFactor(z) for z in Z_TEST])

    np.testing.assert_allclose(D_ours, D_colossus, rtol=RTOL,
                               err_msg="D(z) does not match Colossus")


def test_growth_factor_derivative():
    """Test dD/dz against numerical derivative of Colossus D(z)."""
    cosmo = setup_colossus_cosmology()

    # Our implementation
    dD_ours = dD_dz(TEST_COSMO, Z_TEST)

    # Numerical derivative of Colossus
    def D_colossus_func(z):
        return cosmo.growthFactor(z)

    dD_numerical = five_point_derivative(D_colossus_func, Z_TEST)

    np.testing.assert_allclose(dD_ours, dD_numerical, rtol=RTOL,
                               err_msg="dD/dz does not match numerical derivative")


def test_multiple_cosmologies():
    """Test across different regions of w0-wa parameter space."""
    test_cases = [
        {"w0": -1.0, "wa": 0.0},   # ΛCDM
        {"w0": -0.9, "wa": 0.0},   # Constant w
        {"w0": -0.9, "wa": 0.2},   # Evolving w
        {"w0": -1.1, "wa": 0.3},   # Phantom-like
    ]

    z_test = np.array([0.0, 0.5, 1.0, 2.0])

    for case in test_cases:
        cosmo_params = TEST_COSMO.copy()
        cosmo_params.update(case)

        # Setup Colossus
        params = {
            'flat': True,
            'H0': cosmo_params['hubble'] * 100,
            'Om0': cosmo_params['omega_cold'],
            'Ob0': cosmo_params['omega_baryon'],
            'sigma8': cosmo_params['sigma8_cold'],
            'ns': cosmo_params['ns'],
            'de_model': 'w0wa',
            'w0': cosmo_params['w0'],
            'wa': cosmo_params['wa']
        }
        cosmology.addCosmology(f'test_{case["w0"]}_{case["wa"]}', params)
        cosmo = cosmology.setCosmology(f'test_{case["w0"]}_{case["wa"]}')

        # Test H(z)
        H_ours = H(cosmo_params, z_test)
        H_colossus = np.array([cosmo.Hz(z) for z in z_test])
        np.testing.assert_allclose(H_ours, H_colossus, rtol=RTOL,
                                   err_msg=f"H(z) failed for w0={case['w0']}, wa={case['wa']}")

        # Test D(z)
        D_ours = D(cosmo_params, z_test)
        D_colossus = np.array([cosmo.growthFactor(z) for z in z_test])
        np.testing.assert_allclose(D_ours, D_colossus, rtol=RTOL,
                                   err_msg=f"D(z) failed for w0={case['w0']}, wa={case['wa']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
