"""
tremor_analysis.py

Extracts tremor biomarkers from a 1D motion signal.
Designed for webcam-based wrist tracking.
"""

import numpy as np
from scipy.fft import fft


# ===============================
# Basic Signal Metrics
# ===============================

def calculate_variance(signal):
    return np.var(signal)


def calculate_amplitude(signal):
    return np.max(signal) - np.min(signal)


# ===============================
# Dominant Frequency via FFT
# ===============================

def dominant_frequency(signal, sampling_rate):
    n = len(signal)

    # FFT
    fft_values = np.abs(fft(signal))
    freqs = np.fft.fftfreq(n, 1 / sampling_rate)

    # Use only positive frequencies
    positive_freqs = freqs[:n // 2]
    positive_fft = fft_values[:n // 2]

    # Ignore very low frequencies (< 0.5 Hz)
    mask = positive_freqs > 0.5
    positive_freqs = positive_freqs[mask]
    positive_fft = positive_fft[mask]

    if len(positive_fft) == 0:
        return 0.0

    peak_freq = positive_freqs[np.argmax(positive_fft)]

    return float(abs(peak_freq))


# ===============================
# Tremor Scoring Logic
# ===============================

def tremor_score(signal, sampling_rate):
    var = calculate_variance(signal)
    amp = calculate_amplitude(signal)
    freq = dominant_frequency(signal, sampling_rate)

    score = 0

    # Slightly widened tremor band for demo robustness
    if 3 <= freq <= 7:
        score += 3

    # Only add instability score if oscillatory frequency detected
    if score > 0 and var > 5:
        score += 1

    return {
        "variance": float(var),
        "amplitude": float(amp),
        "dominant_frequency": float(freq),
        "tremor_score": int(score)
    }