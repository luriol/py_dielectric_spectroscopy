
import numpy as np
import struct
from scipy.optimize import root_scalar
import zipfile


import numpy as np
from scipy.optimize import root_scalar




def pt1000_lookup(R_measured):
    """
    Given resistance R in ohms, return the corresponding temperature in °C
    for a Pt1000 using the inverse of the Callendar–Van Dusen equation.
    Returns -999.0 on any error or out-of-range input.
    """
    # Constants for Pt1000 (ITS-90)
    R0 = 1000.0
    A = 3.9083e-3
    B = -5.775e-7
    C_neg = -4.183e-12  # Only used for T < 0 °C

    # Quick input guards
    if R_measured is None or not np.isfinite(R_measured):
        return -999.0

    def R_of_T(T):
        C = C_neg if T < 0 else 0.0
        return R0 * (1 + A*T + B*T*T + C*(T - 100.0)*T**3)

    # Physical validity check for the given bracket
    Tmin, Tmax = -200.0, 850.0
    Rmin, Rmax = R_of_T(Tmin), R_of_T(Tmax)
    lo, hi = (min(Rmin, Rmax), max(Rmin, Rmax))
    if not (lo <= R_measured <= hi):
        return -999.0

    try:
        # Use analytic inverse for T >= 0: R = R0 * (1 + A T + B T^2)
        if R_measured >= R0:
            # Solve B T^2 + A T + (1 - R/R0) = 0
            c = 1.0 - (R_measured / R0)
            disc = A*A - 4.0*B*c
            if disc < 0.0:
                return -999.0
            # Pick the physically meaningful root (positive T, B < 0)
            T1 = (-A + np.sqrt(disc)) / (2.0*B)
            T2 = (-A - np.sqrt(disc)) / (2.0*B)
            # Choose the root ≥ 0 and within [0, Tmax]
            candidates = [t for t in (T1, T2) if np.isfinite(t) and (0.0 <= t <= Tmax)]
            if candidates:
                return float(candidates[0])
            # Fallback to numeric if analytic selection failed
            sol = root_scalar(lambda T: R_of_T(T) - R_measured,
                              bracket=[0.0, Tmax], method='brentq')
            return float(sol.root) if sol.converged else -999.0

        # For T < 0, use full CVD with C term and brentq in [-200, 0]
        sol = root_scalar(lambda T: R_of_T(T) - R_measured,
                          bracket=[Tmin, 0.0], method='brentq')
        return float(sol.root) if sol.converged else -999.0

    except Exception:
        return -999.0

    
def parse_teensy_bin(raw):
    V_REF       = 3.3
    ADC_MAX_10  = 1023.0
    ADC_MAX_12  = 4095.0
    R_REF       = 1000.0

    S_HIGH      = 50
    S_LOW       = 16000

    BYTES_H     = S_HIGH * 2
    BYTES_TH    = 4
    BYTES_L     = S_LOW * 2
    BYTES_TL1   = 4
    BYTES_TL2   = 4
    BYTES_AG    = 4
    TOTAL_BYTES = BYTES_H + BYTES_TH + BYTES_L + BYTES_TL1 + BYTES_TL2 + BYTES_AG
    idx = 0
    vh        = np.frombuffer(raw[idx:idx+BYTES_H],   dtype=np.uint16); idx += BYTES_H
    t_high    = struct.unpack('<I',  raw[idx:idx+BYTES_TH])[0];       idx += BYTES_TH
    vl        = np.frombuffer(raw[idx:idx+BYTES_L],   dtype=np.uint16); idx += BYTES_L
    totalLow1 = struct.unpack('<I',  raw[idx:idx+BYTES_TL1])[0];      idx += BYTES_TL1
    totalLow  = struct.unpack('<I',  raw[idx:idx+BYTES_TL2])[0];      idx += BYTES_TL2
    avg_count = struct.unpack('<f',  raw[idx:idx+BYTES_AG])[0]        # float32

    # Compute timing per sample:
    dt_h = t_high / S_HIGH
    dt_l1 = totalLow1 / 1200.0         # First 1200 low-speed samples
    dt_l2 = (totalLow - totalLow1) / (S_LOW - 1200.0)  # Remaining low-speed samples
    toff = 2.43
    t_h = np.arange(S_HIGH) * dt_h
    t_l1 = t_h[-1] + dt_h + np.arange(1200) * dt_l1 + toff
    t_l2 = t_l1[-1] + dt_l1 + np.arange(S_LOW - 1200) * dt_l2 + toff

    # Convert ADC counts to voltages
    v_h = vh * (V_REF / ADC_MAX_10)
    v_l1 = vl[:1200] * (V_REF / ADC_MAX_12)
    v_l2 = vl[1200:] * (V_REF / ADC_MAX_12)

    # Calculate temperature from thermistor ADC average count
    V_th = avg_count / ADC_MAX_10 * V_REF
    if V_th == 0 or V_th >= V_REF:
        T_C = None
    else:
        R_th = R_REF * V_th / (V_REF - V_th)
        T_C = pt1000_lookup(R_th)

    return t_h, v_h, t_l1, v_l1, t_l2, v_l2, T_C, R_th 

import re

def extract_number(fname):
    """Extract the last integer in the filename (e.g., ..._10.bin → 10)."""
    matches = re.findall(r'(\d+)', fname)
    return int(matches[-1]) if matches else float('inf')


# Sort by numeric index in filename

def load_data_files(zip_path, verbose=True):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Only process actual files (no directories)
        bin_files = sorted(
            [f for f in zip_ref.namelist() if not f.endswith('/')],
            key=extract_number
        )
        all_times = []
        all_voltages = []
        all_temperatures = []
        all_R_th = []
        vdata = {}

        if verbose:
            print(f"[INFO] Found {len(bin_files)} files in {zip_path.name}")

        for fname in bin_files:
            info = zip_ref.getinfo(fname)
            if info.file_size == 0:
                if verbose:
                    print(f"[WARN] Skipping empty file: {fname}")
                continue

            if verbose:
                print(f"[INFO] Reading: {fname}  ({info.file_size} bytes)")

            try:
                with zip_ref.open(fname) as f:
                    raw = f.read()

                # Parse the binary data block
                result = parse_teensy_bin(raw)
                if result[0] is None:
                    print(f"[WARN] Skipping corrupted file: {fname}")
                    continue

                t_h, v_h, t_l1, v_l1, t_l2, v_l2, T_C, R_th = result

                # Concatenate all time and voltage segments
                t_all = np.concatenate((t_h, t_l1, t_l2))
                v_all = np.concatenate((v_h, v_l1, v_l2))

                all_times.append(t_all)
                all_voltages.append(v_all)
                all_temperatures.append(T_C)
                all_R_th.append(R_th)

                if verbose and T_C is not None:
                    print(f"  ✓ Parsed OK — T = {T_C:.2f} °C")

            except Exception as e:
                print(f"[ERROR] Failed to parse {fname}: {e}")
                continue

        vdata['times'] = all_times
        vdata['voltages'] = all_voltages
        vdata['temperatures'] = all_temperatures
        vdata['R_th'] = all_R_th

        if verbose:
            print(f"[INFO] Loaded {len(all_times)} valid datasets from {zip_path.name}")

    return vdata

def bin_by_t(vdata,delta_T):
    binned = {}
    times = vdata['times']
    voltages = vdata['voltages']
    temperatures = vdata['temperatures']
    for t, v, T in zip(times, voltages, temperatures):
        T_bin = delta_T * round(T / delta_T)
        binned[T_bin] = {'times': [], 'voltages': [], 'temps': []}
        binned[T_bin]['times'].append(t)
        binned[T_bin]['voltages'].append(v)
        binned[T_bin]['temps'].append(T)
    results = []
    for T_bin in sorted(binned.keys()):
        times_stack = np.stack(binned[T_bin]['times'])
        volts_stack = np.stack(binned[T_bin]['voltages'])
        temps = binned[T_bin]['temps']
        t_avg = np.mean(times_stack, axis=0)
        v_avg = np.mean(volts_stack, axis=0)
        T_avg = np.mean(temps)
        results.append((t_avg, v_avg, T_avg))
    return(results)

import numpy as np

def log_gaussian_smooth(x, y, rel_half_width=0.10, num=None):
    """
    Smooth y(x) with a Gaussian whose window is constant in log(x).

    Parameters
    ----------
    x : array_like
        Strictly positive x-values (monotonic not required).
    y : array_like
        Values to smooth, same length as x.
    rel_half_width : float
        Fractional half-width of the window in x, i.e. use points within
        [x*(1-rel_half_width), x*(1+rel_half_width)] approximately.
        Internally this is implemented as a Gaussian of half-width
        h = ln(1 + rel_half_width) in log-space (symmetric in log).
    num : int or None
        Number of points for a uniform log grid. Defaults to len(x).

    Returns
    -------
    y_smooth : ndarray
        Smoothed y evaluated at the original x positions.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.any(x <= 0):
        raise ValueError("x must be strictly positive for log smoothing.")

    # 1) sort by x
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # 2) go to log-space
    u = np.log(x)

    # 3) resample onto a uniform u-grid
    if num is None:
        num = len(x)
    u_grid = np.linspace(u.min(), u.max(), num)
    y_grid = np.interp(u_grid, u, y)

    # 4) build Gaussian kernel in grid units
    h = np.log(1.0 + rel_half_width)         # half-width in log-units
    du = np.mean(np.diff(u_grid))            # grid spacing in log-units
    sigma = max(h / du, 1e-12)               # std dev in grid steps
    radius = int(np.ceil(4 * sigma))         # truncate at ~±4σ
    k = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (k / sigma) ** 2)
    kernel /= kernel.sum()

    # 5) convolve on the uniform log grid (reflect-ish edges)
    # pad by reflection to reduce edge bias
    pad_left = y_grid[1:radius+1][::-1]
    pad_right = y_grid[-radius-1:-1][::-1]
    y_padded = np.concatenate([pad_left, y_grid, pad_right])
    y_sm_grid = np.convolve(y_padded, kernel, mode='same')[radius:-radius]

    # 6) map back to original (unsorted) x locations
    y_sm_at_u = np.interp(u, u_grid, y_sm_grid)
    y_smooth = np.empty_like(y_sm_at_u)
    y_smooth[order] = y_sm_at_u
    return y_smooth
