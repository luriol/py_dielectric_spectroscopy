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


import numpy as np
import scipy.constants as scc

import numpy as np
import scipy.constants as scc

def _debye_sigma_poles_residues(R, C0, k_inf, Delta_k, tau, rho=None):
    C_inf = k_inf * C0
    dC    = Delta_k * C0
    G = 0.0 if (rho is None or not np.isfinite(rho) or rho <= 0.0) else C0/(rho*scc.epsilon_0)

    a = R * C_inf * tau
    b = tau * (1.0 + R * G) + R * (C_inf + dC)
    c = 1.0 + R * G

    disc = b*b - 4.0*a*c
    if disc < 0:  # numeric guard
        disc = 0.0
    sqrt_disc = np.sqrt(disc)

    s1 = (-b + sqrt_disc) / (2.0 * a)
    s2 = (-b - sqrt_disc) / (2.0 * a)

    # residues for unit step of H(s)/s
    B = (1.0 + s1*tau) / (a * s1 * (s1 - s2))
    C = (1.0 + s2*tau) / (a * s2 * (s2 - s1))

    V_inf = 1.0 / c  # DC gain = 1/(1+RG)

    return s1, s2, B, C, V_inf


def V_debye_sigma_falling_edge(t, R, C0, k_inf, Delta_k, tau, rho=None):
    """
    Analytic node voltage for a *falling* input step: Vin switches high->low at t=0.
    Output starts at v(0+)=V_inf and decays to 0:
        v(t) = -B*exp(s1*t) - C*exp(s2*t),  t >= 0
        v(t) = V_inf for t < 0  (previous steady high level; not returned)

    With rho -> ∞: V_inf = 1, so v goes 1 -> 0.
    With finite rho: V_inf = 1/(1+R*C0/(rho*ε0)) < 1, so v goes (<1) -> 0.

    Parameters
    ----------
    t : array_like (s)
    R, C0, k_inf, Delta_k, tau, rho : floats
        Circuit and Debye(+σ) parameters. rho in Ω·m (None/inf => no conductivity).

    Returns
    -------
    v : ndarray
        Node voltage for t >= 0; zero for t < 0 is not enforced (we only compute the decay).
    """
    t = np.asarray(t, float)
    s1, s2, B, C, V_inf = _debye_sigma_poles_residues(R, C0, k_inf, Delta_k, tau, rho)

    v = np.zeros_like(t, float)
    m = t >= 0.0
    dt = t[m]
    v[m] = -B*np.exp(s1*dt) - C*np.exp(s2*dt)  # starts at V_inf, ends at 0
    return v
