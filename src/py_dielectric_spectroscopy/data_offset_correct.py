import numpy as np
from typing import Union
from scipy.special import erf
from scipy.optimize import brentq

ArrayLike = Union[float, np.ndarray]

def expected_adc_reading(x0: ArrayLike, sigma: float) -> np.ndarray:
    """
    Forward model: expected ADC reading when negatives are clipped to 0 and
    additive noise is Gaussian N(0, sigma^2) on a true signal x0.

    Parameters
    ----------
    x0 : float or array-like
        True (pre-ADC) signal value(s).
    sigma : float
        Standard deviation of zero-mean Gaussian noise (same units as x0).

    Returns
    -------
    y : np.ndarray
        E[max(0, x0 + N(0, sigma^2))] with the same shape as x0.
    """
    x0 = np.asarray(x0, dtype=float)
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    if sigma == 0.0:
        # No noise: clipping only
        return np.maximum(x0, 0.0)

    z = x0 / sigma
    return 0.5 * x0 * (1.0 + erf(z / np.sqrt(2.0))) + (sigma / np.sqrt(2.0*np.pi)) * np.exp(-0.5 * z * z)


def invert_adc_bias(
    y: ArrayLike,
    sigma: float,
    *,
    assume_nonnegative: bool = True,
    tol: float = 1e-12,
    maxiter: int = 100
) -> np.ndarray:
    """
    Invert the rectified-Gaussian bias to estimate the true signal x0 from the
    expected (or averaged) ADC reading y.

    This solves, for each element independently,
        expected_adc_reading(x0, sigma) = y
    using a safe bracketed root finder (brentq). The forward map is strictly
    increasing in x0, so a unique solution exists for any y >= 0.

    Parameters
    ----------
    y : float or array-like
        Observed mean ADC reading(s). Should be >= 0 since ADC clips negatives.
        If you have raw samples, average them first to reduce variance.
    sigma : float
        Standard deviation of the (pre-clip) Gaussian noise.
    assume_nonnegative : bool, default True
        If True, constrain the solution to x0 >= 0 (common for physical signals).
        If False, allow x0 < 0 (e.g., if small negative baselines are possible).
    tol : float, default 1e-12
        Absolute root-finding tolerance.
    maxiter : int, default 100
        Maximum iterations for the root finder.

    Returns
    -------
    x0_hat : np.ndarray
        Estimated true signal(s) with the same shape as y.

    Notes
    -----
    * High-SNR shortcut: when y >> sigma (≈ 5σ and up), the bias is negligible
      and x0 ≈ y. We detect that regime and return y directly for speed.
    * Edge cases:
        - sigma == 0 ⇒ x0_hat = y (since only clipping remains).
        - y == 0 ⇒ x0_hat ≤ 0 if assume_nonnegative=False, else x0_hat = 0.
    """
    y = np.asarray(y, dtype=float)
    if np.any(y < 0):
        raise ValueError("y should be nonnegative (ADC outputs are clipped at 0).")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")

    # Handle trivial cases fast
    if sigma == 0.0:
        return np.array(y, copy=True)
    # Bias is negligible when y is well above zero relative to noise
    hi_snr_mask = y >= (5.0 * sigma)
    out = np.empty_like(y)
    out[hi_snr_mask] = y[hi_snr_mask]

    # Solve the rest with brentq
    to_solve_mask = ~hi_snr_mask
    if not np.any(to_solve_mask):
        return out

    y_solve = y[to_solve_mask]

    # Element-wise solver with robust bracketing
    def solve_one(y_i: float) -> float:
        # f(x0) = E[max(0, x0 + N(0,σ²))] - y_i; find root in x0
        def f(x0: float) -> float:
            return expected_adc_reading(x0, sigma) - y_i

        # Lower bound:
        if assume_nonnegative:
            lo = 0.0
        else:
            # Start below zero, expand if needed
            lo = -10.0 * sigma

        # Upper bound: must be >= y_i (since for large x0, bias→0 and E≈x0)
        hi = max(y_i, 0.0) + 2.0 * sigma

        # Expand bounds until f(lo) <= 0 <= f(hi)
        flo = f(lo)
        # If assume_nonnegative and y_i==0, flo may already be 0 at lo=0
        if flo > 0:
            # move lo downward until f(lo) <= 0
            k = 1
            while flo > 0 and k <= 20:
                lo -= (2.0 ** k) * sigma
                flo = f(lo)
                k += 1

        fhi = f(hi)
        k = 1
        while fhi < 0 and k <= 20:
            hi += (2.0 ** k) * sigma
            fhi = f(hi)
            k += 1

        # As a last resort, if y_i==0 and assume_nonnegative, the solution is 0.
        if y_i == 0.0 and assume_nonnegative:
            return 0.0

        return brentq(f, lo, hi, xtol=tol, maxiter=maxiter)

    out[to_solve_mask] = np.vectorize(solve_one, otypes=[float])(y_solve)
    return out