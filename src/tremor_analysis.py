import numpy as np
from scipy.fft import fft


def calculate_variance(signal):
    return np.var(signal)


def calculate_amplitude(signal):
    return np.max(signal) - np.min(signal)


def dominant_frequency(signal, sampling_rate):
    fft_values = np.abs(fft(signal))
    freqs = np.fft.fftfreq(len(signal), 1 / sampling_rate)

    positive_freqs = freqs[:len(freqs)//2]
    positive_fft = fft_values[:len(freqs)//2]

    peak_freq = positive_freqs[np.argmax(positive_fft)]
    return abs(peak_freq)


def tremor_score(signal, sampling_rate):
    var = calculate_variance(signal)
    amp = calculate_amplitude(signal)
    freq = dominant_frequency(signal, sampling_rate)

    score = 0

    # Frequency check (Parkinson tremor ~4–6 Hz)
    if 4 <= freq <= 6:
        score += 2

    # Variance threshold
    if var > 1:
        score += 1

    # Amplitude threshold
    if amp > 3:
        score += 1

    return {
        "variance": float(var),
        "amplitude": float(amp),
        "dominant_frequency": float(freq),
        "tremor_score": int(score)
    }