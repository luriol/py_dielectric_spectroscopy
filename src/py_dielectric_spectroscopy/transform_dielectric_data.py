"""Square-wave response and dielectric transfer functions.

The forward model has three layers:
    capacitance model -> RC divider -> odd-harmonic square-wave response.
The inverse/time-to-frequency routines are retained below.
"""

import numpy as np
import scipy.constants as scc


# -----------------------------------------------------------------------------
# Shared numerical and circuit calculations
# -----------------------------------------------------------------------------

def get_NT(t):
    """Return (N, T), assuming t contains N uniformly spaced half-period samples.

    The full square-wave period is T = 2*N*dt. The original code's convention
    is preserved: t is expected to start at zero, with the endpoint excluded.
    """
    N = t.size
    dt = t[1] - t[0]
    T = 2 * (t[-1] + dt)
    return N, T

def square_wave_response(t, transfer_function, dc_gain=1.0,
                         V_high=3.0, V_low=0.0, **kwargs):
    """Physical voltage response to a square wave."""

    N, T = get_NT(t)
    harmonics = 2 * np.arange(N) + 1
    W = 2 * np.pi * harmonics / T

    E = np.exp(1j * np.outer(W, t))
    b = -4 / (W * T)
    J = transfer_function(W, **kwargs)

    v = np.imag(np.dot(b * J, E))

    V_mid = (V_high + V_low) / 2
    V_pp = V_high - V_low

    return V_mid * dc_gain + V_pp * v



# -----------------------------------------------------------------------------
# 2. Debye dielectric with DC conductivity
# -----------------------------------------------------------------------------

def V_debye_sim(t, R, C0, k0, Delta_k, tau, rho,
                V_high=3.0, V_low=0.0):
    """Voltage response of a Debye dielectric following a falling square-wave edge."""

    def transfer_function(W):
        kappa = (
            k0
            + Delta_k / (1 + 1j * W * tau)
            - 1j / (rho * scc.epsilon_0 * W)
        )
        return 1 / (1 + 1j * W * R * C0 * kappa)

    dc_gain = 1 / (1 + R * C0 / (rho * scc.epsilon_0))

    return square_wave_response(
        t,
        transfer_function,
        dc_gain=dc_gain,
        V_high=V_high,
        V_low=V_low
    )

# -----------------------------------------------------------------------------
# 3. Cole-Cole dielectric with DC conductivity
# -----------------------------------------------------------------------------

def V_cole_cole_sim(t, R, C0, k_inf, Delta_k, tau, alpha, rho,
                    V_high=3.0, V_low=0.0):
    """Voltage response of a Cole-Cole dielectric following a falling square-wave edge."""

    def transfer_function(W):
        kappa = (
            k_inf
            + Delta_k / (1 + (1j * W * tau)**alpha)
            - 1j / (rho * scc.epsilon_0 * W)
        )
        return 1 / (1 + 1j * W * R * C0 * kappa)

    dc_gain = 1 / (1 + R * C0 / (rho * scc.epsilon_0))

    return square_wave_response(
        t,
        transfer_function,
        dc_gain=dc_gain,
        V_high=V_high,
        V_low=V_low
    )


# -----------------------------------------------------------------------------
# Existing inverse/time-to-frequency functions (unchanged)
# -----------------------------------------------------------------------------

def get_J(t, Vt):
    """
    Compute the frequency-dependent transfer function J(ω) from measured or simulated voltage.

    Parameters:
        t (ndarray): Time array covering half of the square-wave period (seconds).
        Vt (ndarray): Measured or simulated voltage across the capacitor (V).

    Returns:
        tuple:
            - W (ndarray): Angular frequencies (rad/s).
            - J (ndarray): Frequency-dependent complex transfer function.
    """
    # First normalize and offset Vt to have zero mean
    Vt = Vt/Vt[0] - .5
    N, T = get_NT(t)
    indcs = 2 * np.arange(N) + 1                   # Odd harmonic indices
    W = 2 * np.pi * indcs / T                      # Corresponding angular frequencies
    E = np.exp(-1j * np.outer(W, t))               # Negative exponential for inverse Fourier transform
    b = -4 / (W * T)                               # Fourier coefficients (square wave)
    J = 1j * (4 / (2 * N)) * np.dot(E, Vt) / b     # Inverse transform to retrieve transfer function
    return W[:N // 2], J[:N // 2]                  # Keep positive frequency components only

def odd_fft_transform_aligned(t, Vt):
    """
    FFT-based odd-harmonic transform with exact normalization
    and phase alignment to match the direct O(N^2) implementation.
    """
    # Preprocess exactly like your pipeline
    Vt = Vt / Vt[0] - 0.5

    N = Vt.size
    t0 = t[0]
    # Use endpoint dt to avoid cumulative float error
    dt = (t[-1] - t[0]) / (N - 1)
    # Full-period consistent with FFT of mirrored array
    T = 2 * N * dt

    # Enforce antisymmetric extension for odd-only spectrum
    Vfull = np.concatenate([Vt, -Vt])  # length 2N

    # Unnormalized FFT (exact DFT)
    F = np.fft.fft(Vfull)

    # Odd harmonic bins and angular frequencies
    odd_bins = 2 * np.arange(N) + 1            # 1,3,5,...,(2N-1)
    W = 2 * np.pi * odd_bins / T               # matches direct method grid

    # Scaling (from earlier derivation): J_n = -(i * W * T)/(4N) * F[odd_bin]
    # Phase-align to the true start time t0:
    phase = np.exp(-1j * W * t0)
    J = -1j * (W * T) * F[odd_bins] * phase / (4 * N)

    # Keep positive-frequency half, like the original
    return W[:N // 2], J[:N // 2]




def get_kappa(t, Vt, RC):
    W, J = odd_fft_transform_aligned(t, Vt)

    # --- FIX intermittent off-by-one ---
    Nmin = min(len(W), len(J))
    if len(W) != len(J):
        print(f"[WARN] odd_fft length mismatch: W={len(W)}, J={len(J)} → trimming to {Nmin}")

    W = W[:Nmin]
    J = J[:Nmin]

    kappa = (1 / J - 1) / (1j * W * RC)
    return W, kappa


def V_two_cole_cole_sim(t, R, C0, k_inf,
                        Delta_k1, tau1, alpha1,
                        Delta_k2, tau2, alpha2, rho,
                        V_high=3.0, V_low=0.0):
    """Voltage response for two Cole-Cole relaxations and DC conductivity."""

    def transfer_function(W):
        kappa = (
            k_inf
            + Delta_k1 / (1 + (1j * W * tau1)**alpha1)
            + Delta_k2 / (1 + (1j * W * tau2)**alpha2)
            - 1j / (rho * scc.epsilon_0 * W)
        )
        return 1 / (1 + 1j * W * R * C0 * kappa)

    dc_gain = 1 / (1 + R * C0 / (rho * scc.epsilon_0))

    return square_wave_response(
        t, transfer_function,
        dc_gain=dc_gain,
        V_high=V_high,
        V_low=V_low
    )
