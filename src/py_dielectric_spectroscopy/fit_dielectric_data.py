import scipy.constants as scc
import numpy as np

def get_NT(t):
    """
    Determine number of data points and full period T of the square wave.

    Assumes input array t covers exactly half of the square-wave period.

    Parameters:
        t (ndarray): Time array covering half a period (seconds).

    Returns:
        tuple:
            - N (int): Number of points in the half-period.
            - T (float): Full period of the square wave (seconds).
    """
    N = t.size
    dt = t[1] - t[0]
    T = 2 * (t[-1] + dt)
    return N, T

# ---------- dielectric models ----------
def sim_kappa_cole_cole(W, k_inf, Delta_k, tau, alpha, rho=None, n=1.0):
    """
    Cole–Cole dielectric with optional power-law conduction tail.
    κ(ω) = k_inf + Δk / (1 + (j ω τ)^α)  -  j / (ρ ε0 ω^n),  0 < α ≤ 1
    """
    W = np.asarray(W, dtype=float)
    kappa = k_inf + Delta_k / (1.0 + (1j*W*tau)**alpha)
    if (rho is not None) and np.isfinite(rho):
        kappa = kappa - 1j/(rho * scc.epsilon_0 * (W**float(n)))
    return kappa



def V_cole_cole_sim(t, R, C0, k_inf, Delta_k, tau, alpha, rho, n=1.0, harmonics=None):
    """Same as above but with Cole–Cole κ."""
    N, T = get_NT(t)
    H = int(harmonics) if harmonics is not None else N
    h_idx = 2*np.arange(H) + 1
    W = 2*np.pi * h_idx / T

    E = np.exp(1j * np.outer(W, t - t[0]))
    b = -4.0 / (W * T)

    Cw = C0 * sim_kappa_cole_cole(W, k_inf, Delta_k, tau, alpha, rho=rho, n=n)
    Jw = 1.0 / (1.0 + 1j * W * R * Cw)

    Vt = np.imag((b * Jw) @ E)
    return Vt + Vt[0]

def V_cole_cole_sim_I(t, R, C0, k_inf, Delta_k, tau, alpha, rho, CI, n=1.0, harmonics=None):
    """
    Cole–Cole simulation with *two identical* insulating films (capacitance CI each).
    Tap is between ZI and Zw, so Jw = (ZI + Zw) / (R + 2*ZI + Zw).
    """
    # --- frequency grid (odd harmonics) ---
    N, T = get_NT(t)
    H = int(harmonics) if harmonics is not None else N
    h_idx = 2*np.arange(H) + 1
    W = 2*np.pi * h_idx / T  # shape (H,)

    # --- Fourier basis for square-wave reconstruction ---
    E = np.exp(1j * np.outer(W, t - t[0]))  # (H, len(t))
    b = -4.0 / (W * T)                      # (H,)

    # --- sample dielectric (Cole–Cole) as an effective capacitance Cw(ω) ---
    Cw = C0 * sim_kappa_cole_cole(W, k_inf, Delta_k, tau, alpha, rho=rho, n=n)  # (H,)

    # --- impedances ---
    ZI = 1.0 / (1j * W * CI)   # identical top/bottom films
    Zw = 1.0 / (1j * W * Cw)

    # --- divider transfer function at the tap ---
    Jw = (ZI + Zw) / (R + 2*ZI + Zw)        # (H,)

    # --- synthesize time signal (imag part of sine series) ---
    Vt = np.imag((b * Jw) @ E)              # (len(t),)
    return Vt + Vt[0]



