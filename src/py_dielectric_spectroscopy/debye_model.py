import numpy as np
import scipy.constants as scc

def debye_kappa(W, k_inf, Delta_k, tau, rho):
    """Calculate the complex dielectric function κ(ω) for a Debye dielectric with conductivity."""
    kappa = k_inf + Delta_k / (1 + 1j * W * tau) - 1j / (rho * scc.epsilon_0 * W)
    return kappa

def debye_frequency(W,R,C0,k_inf,Delta_k,tau,rho):
    """Calculate the Debye relaxation in frequency space."""
    kappa =  debye_kappa(W, k_inf, Delta_k, tau, rho)
    V = 1 / (1 + 1j * W * R * C0 * kappa) 
    return V


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

def V_debye_time(t,R,C0,k_inf,Delta_k,tau,rho):
    tau_inf = k_inf*R*C0
    tau_s = (Delta_k +k_inf)*R*C0 
    tau_sig = rho*scc.epsilon_0
    tau_0 = R*C0 

    C = tau*(tau_0 + tau_sig)  
    A = tau_inf*tau_sig 
    B = tau_s*tau_sig + C

    alpha_1 = (-B + np.sqrt(B**2 - 4*A*C +0j))/(2*A)
    alpha_2 = (-B - np.sqrt(B**2 - 4*A*C+0j))/(2*A)

    V = (1 + alpha_1)*np.exp(alpha_1*t/tau)/alpha_1/(alpha_1-alpha_2)
    V += (1 + alpha_2)*np.exp(alpha_2*t/tau)/alpha_2/(alpha_2-alpha_1)
    V *= tau/tau_inf
    V += tau/2/alpha_1/alpha_2/tau_inf 
    return V.real +.5


def V_debye_time_stable(t, R, C0, k_inf, Delta_k, tau, rho):
    """
    More numerically stable version of V_debye_time.
    Returns values intended to be in [0, 1] (optionally clipped).
    """
    clip01=True
    eps_root=1e-30
    eps_equal=1e-12
    t = np.asarray(t, dtype=float)

    # Basic times
    tau_inf = k_inf * R * C0
    tau_s   = (Delta_k + k_inf) * R * C0
    tau_sig = rho * scc.epsilon_0
    tau_0   = R * C0

    # Quadratic coefficients in alpha:
    # A*alpha^2 + B*alpha + C = 0
    C = tau * (tau_0 + tau_sig)
    A = tau_inf * tau_sig
    B = tau_s * tau_sig + C

    # If A is ~0, the quadratic degenerates; avoid division by ~0.
    # (In that regime your model likely needs a separate limiting form.)
    if np.any(np.abs(A) < eps_root):
        V = np.full_like(t, np.nan, dtype=float)
        if clip01:
            return np.clip(np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
        return np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)

    # Work in complex to keep sqrt well-defined for negative discriminant
    A = A + 0j
    B = B + 0j
    Cc = C + 0j

    D = B*B - 4*A*Cc
    sqrtD = np.sqrt(D)

    # Stable quadratic root computation:
    # q = -0.5*(B + sign(B)*sqrtD)  (for real B)
    # For complex, use a phase-based "sign" to avoid zero division.
    signB = np.where(np.abs(B) > 0, B/np.abs(B), 1.0 + 0j)
    q = -0.5 * (B + signB*sqrtD)

    # Roots: alpha1 = q/A, alpha2 = C/q
    # Guard q ~ 0
    if np.any(np.abs(q) < eps_root):
        # Fall back to the direct formula if q is too small
        alpha1 = (-B + sqrtD) / (2*A)
        alpha2 = (-B - sqrtD) / (2*A)
    else:
        alpha1 = q / A
        alpha2 = Cc / q

    x = t / tau  # dimensionless time

    # f(alpha) = ((1+alpha)/alpha) * exp(alpha*x)
    # Evaluate as written; the *difference quotient* below avoids the worst singularities.
    def f(alpha):
        return ((1.0 + alpha) / alpha) * np.exp(alpha * x)

    # When alpha1 ~ alpha2, use derivative limit: V ~ (tau/tau_inf) * f'(alpha)
    # f'(a) = exp(a x) * [ x*(1+1/a) - 1/a^2 ]
    def fprime(alpha):
        ea = np.exp(alpha * x)
        return ea * (x * (1.0 + 1.0/alpha) - 1.0/(alpha*alpha))

    da = alpha1 - alpha2
    use_deriv = np.abs(da) <= eps_equal * (1.0 + np.maximum(np.abs(alpha1), np.abs(alpha2)))

    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        Vc = np.empty_like(x, dtype=complex)
        if np.isscalar(use_deriv):
            if use_deriv:
                alpha = -B/(2*A)  # double-root location
                Vc = fprime(alpha)
            else:
                Vc = (f(alpha1) - f(alpha2)) / da
        else:
            # vectorized mask (rare here since alpha1/alpha2 are scalars, but safe)
            alpha = -B/(2*A)
            Vc[use_deriv] = fprime(alpha)
            Vc[~use_deriv] = (f(alpha1) - f(alpha2)) / da

        Vc *= (tau / tau_inf)
    
    Vc +=     tau/2/alpha1/alpha2/tau_inf 
    V = -Vc.real
    

    # Last-resort cleanup
    V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)

    if clip01:
        V = np.clip(V, 0.0, 1.0)

    return V