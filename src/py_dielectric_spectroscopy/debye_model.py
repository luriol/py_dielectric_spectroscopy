
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
