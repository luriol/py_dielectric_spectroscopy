import numpy as np
import scipy.constants as scc

def _debye_sigma_poles_residues(R, C0, k_inf, Delta_k, tau, rho=None):
    """Compute complex poles and residues for the Debye + conductivity model."""
    params = np.array([R, C0, k_inf, Delta_k, tau], float)
    if np.any(np.isnan(params)) or tau <= 0 or C0 <= 0 or R <= 0:
        return np.nan + 0j, np.nan + 0j, np.nan + 0j, np.nan + 0j, np.nan

    C_inf = k_inf * C0
    dC = Delta_k * C0
    G = 0.0 if (rho is None or not np.isfinite(rho) or rho <= 0.0) else C0 / (rho * scc.epsilon_0)

    a = R * C_inf * tau
    b = tau * (1.0 + R * G) + R * (C_inf + dC)
    c = 1.0 + R * G

    disc = b*b - 4.0*a*c + 0j
    sqrt_disc = np.sqrt(disc)

    eps = 1e-18
    if abs(a) < eps:
        return np.nan + 0j, np.nan + 0j, np.nan + 0j, np.nan + 0j, np.nan

    s1 = (-b + sqrt_disc) / (2.0 * a)
    s2 = (-b - sqrt_disc) / (2.0 * a)

    denom1 = a * s1 * (s1 - s2)
    denom2 = a * s2 * (s2 - s1)
    if abs(denom1) < eps or abs(denom2) < eps:
        return np.nan + 0j, np.nan + 0j, np.nan + 0j, np.nan + 0j, np.nan

    B = (1.0 + s1 * tau) / denom1
    C = (1.0 + s2 * tau) / denom2
    V_inf = 1.0 / c if abs(c) > eps else np.nan

    return s1, s2, B, C, V_inf


def V_debye_sigma_falling_edge(t, R, C0, k_inf, Delta_k, tau, rho=None):
    """Analytic node voltage for a falling input step (real part only)."""
    t = np.asarray(t, float)
    s1, s2, B, C, V_inf = _debye_sigma_poles_residues(R, C0, k_inf, Delta_k, tau, rho)

    if np.any(np.isnan([np.real(s1), np.real(s2), np.real(B), np.real(C), V_inf])):
        return np.full_like(t, np.nan, dtype=float)

    v = np.zeros_like(t, dtype=complex)
    m = t >= 0
    dt = t[m]

    # guard against overflow in exp
    exp1 = np.exp(np.clip(np.real(s1)*dt, -700, 700)) * np.exp(1j*np.imag(s1)*dt)
    exp2 = np.exp(np.clip(np.real(s2)*dt, -700, 700)) * np.exp(1j*np.imag(s2)*dt)
    v[m] = -B * exp1 - C * exp2

    return np.real(v)
