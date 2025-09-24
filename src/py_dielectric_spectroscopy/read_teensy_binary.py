import struct
import os
import numpy as np


N_SAMPLES   = 8192
BYTES_PER_ADC_ARRAY = N_SAMPLES * 2
BYTES_TIME  = 4
BYTES_ADC   = BYTES_PER_ADC_ARRAY * 2
TOTAL_BYTES = BYTES_ADC + BYTES_TIME


def pt1000_lookup(R):
    """
    Estimate temperature (°C) from Pt1000 resistance (Ω) using linear interpolation
    based on standard reference values.


    Parameters:
    -----------
    R : float or array-like
        Resistance of the Pt1000 sensor in ohms


    Returns:
    --------
    T : float or ndarray
        Estimated temperature in degrees Celsius
    """
    T_ref = [-90,-80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30]
    R_ref = [643.00,683.25, 723.30, 763.30, 803.10, 842.70, 882.20, 921.60, 
             960.90, 1000.00, 1039.00, 1077.90, 1116.70]


    T = np.interp(R, R_ref, T_ref)
    return T






def get_teensy_binary_data(fp):     
    raw = fp.read()


    cap_adc_data    = raw[:BYTES_PER_ADC_ARRAY]
    therm_adc_data  = raw[BYTES_PER_ADC_ARRAY:2 * BYTES_PER_ADC_ARRAY]
    time_data       = raw[2 * BYTES_PER_ADC_ARRAY:]


    cap_readings    = struct.unpack('<' + 'H' * N_SAMPLES, cap_adc_data)
    therm_readings  = struct.unpack('<' + 'H' * N_SAMPLES, therm_adc_data)
    total_time_us   = struct.unpack('<I', time_data)[0]


    ADC_MAX     = 1023.0
    V_REF       = 3.25
    R_OHMS      = 1000000  # 1 MΩ
    CONFIRM_SAMPLES = 5
    R_REF = 1000.0       # Reference resistor in ohms
    # process the data 
    voltages = np.array(cap_readings) *V_REF / ADC_MAX 
    voltage_therm = np.average(np.array(therm_readings) * V_REF / ADC_MAX)
    R_therm = R_REF*voltage_therm/(V_REF-voltage_therm)
    T_therm = pt1000_lookup(R_therm)
    times = np.linspace(0, total_time_us, N_SAMPLES)




    return times, voltages, T_therm 
