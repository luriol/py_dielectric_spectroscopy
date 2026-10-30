
import numpy as np

from py_dielectric_spectroscopy.transform_dielectric_data import (
    V_C_sim,
    V_debye_sim,
    V_cole_cole_sim,
    square_wave_response,
    RC_divider_transfer,
    constant_capacitance,
)


def test_rc_step_response():
    """Compare square-wave simulation with exact RC relaxation."""
    R = 50e3
    C = 80e-12
    tau = R * C

    # Period much longer than RC relaxation time
    T = 200 * tau
    N = 1000
    t = np.arange(N) * (T / (2 * N))

    v = V_C_sim(t, R, C)

    # The existing Fourier convention produces a falling transition
    # at t=0, approaching its negative plateau.
    v_expected = -0.5 + np.exp(-t / tau)

    # Ignore points near discontinuities and compare the relaxation.
    mask = (t > 0.1 * tau) & (t < 8 * tau)

    np.testing.assert_allclose(
        v[mask], v_expected[mask],
        atol=0.02,
        rtol=0,
    )


def test_cole_cole_reduces_to_debye():
    """Cole-Cole with alpha=1 must equal Debye."""
    t = np.arange(500) * 1e-5

    params = dict(
        R=50e3,
        C0=80e-12,
        Delta_k=42,
        tau=1e-3,
        rho=1e6,
    )

    v_debye = V_debye_sim(t, k0=3, **params)
    v_cole = V_cole_cole_sim(
        t, k_inf=3, alpha=1.0, **params
    )

    np.testing.assert_allclose(
        v_cole, v_debye,
        rtol=1e-12,
        atol=1e-12,
    )


def test_general_fourier_matches_direct_sum():
    """General Fourier routine reproduces an explicit harmonic sum."""
    R = 50e3
    C = 80e-12
    N = 128
    dt = 1e-6
    t = np.arange(N) * dt
    T = 2 * N * dt

    n = 2 * np.arange(N) + 1
    W = 2 * np.pi * n / T
    H = 1 / (1 + 1j * W * R * C)

    expected = np.array([
        np.sum(
            -4 / (W * T)
            * np.imag(H * np.exp(1j * W * ti))
        )
        for ti in t
    ])

    actual = square_wave_response(
        t,
        RC_divider_transfer,
        R=R,
        capacitance_function=constant_capacitance,
        C0=C,
    )

    np.testing.assert_allclose(
        actual, expected,
        rtol=1e-12,
        atol=1e-12,
    )
