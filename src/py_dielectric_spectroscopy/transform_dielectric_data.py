import scipy.constants as scc
import numpy as np
from lmfit import Model
import inspect

# test comment

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

def V_debye_sim(t, R, C0, k0, Delta_k, tau, rho):
    """
    Simulate the voltage response across a capacitor with Debye dielectric driven by a square wave.

    The circuit model is a series RC voltage divider, with the capacitor containing the dielectric medium.

    Parameters:
        t (ndarray): Time array covering half of the square-wave period (seconds).
        R (float): Resistance in series with the capacitor (Ω).
        C0 (float): Capacitance of the empty capacitor (F).
        k0 (float): High-frequency dielectric constant.
        Delta_k (float): Dielectric relaxation strength.
        tau (float): Debye relaxation time (seconds).
        rho (float): Resistivity of the dielectric medium (Ω·m).

    Returns:
        ndarray: Simulated voltage across the capacitor over time t (V).
    """
    N, T = get_NT(t)
    indcs = 2 * np.arange(N) + 1                   # Odd harmonics indices
    W = 2 * np.pi * indcs / T                      # Angular frequencies of odd harmonics
    E = np.exp(1j * np.outer(W, t))                # Fourier basis (complex exponentials)
    b = -4 / (W * T)                               # Square wave Fourier coefficients
    C = C0 * sim_kappa(W, k0, Delta_k, tau, rho)   # Complex capacitance with dielectric
    J = 1 / (1 + 1j * W * R * C)                   # Transfer function of RC divider
    arg = b * J                                    # Adjust Fourier amplitudes by transfer function
    Vt = np.imag(np.dot(arg, E))                   # Voltage response (imaginary part ensures real output)
    return Vt + Vt[0]  # Offset to ensure non-negative voltage

def V_C_sim(t, R, C0):
    """
    Simulate the voltage response across a capacitor 
    with constant dielectric driven by a square wave.

    The circuit model is a series RC voltage divider, with the capacitor containing the dielectric medium.


    Returns:
        ndarray: Simulated voltage across the capacitor over time t (V).
    """
    N, T = get_NT(t)
    indcs = 2 * np.arange(N) + 1                   # Odd harmonics indices
    W = 2 * np.pi * indcs / T                      # Angular frequencies of odd harmonics
    E = np.exp(1j * np.outer(W, t))                # Fourier basis (complex exponentials)
    b = -4 / (W * T)                               # Square wave Fourier coefficients
    C = C0    # Complex capacitance with dielectric
    J = 1 / (1 + 1j * W * R * C)                   # Transfer function of RC divider
    arg = b * J                                    # Adjust Fourier amplitudes by transfer function
    Vt = np.imag(np.dot(arg, E))                   # Voltage response (imaginary part ensures real output)
    return Vt

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


def get_kappa(t, Vt, RC):
    """
    Calculate the dielectric function κ(ω) from voltage response V(t) using the RC divider model.

    Parameters:
        t (ndarray): Time array covering half of the square-wave period (seconds).
        Vt (ndarray): Measured or simulated voltage across the capacitor (V).
        RC (float): Product of series resistance R and empty capacitor capacitance C0 (seconds).

    Returns:
        tuple:
            - W (ndarray): Angular frequencies (rad/s).
            - kappa (ndarray): Complex dielectric function κ(ω).
    """
    W, J = get_J(t, Vt)                                # Compute frequency response
    kappa = (1 / J - 1) / (1j * W * RC)                # Invert RC model to retrieve κ(ω)
    return W, kappa
