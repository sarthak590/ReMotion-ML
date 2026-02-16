import numpy as np
import matplotlib.pyplot as plt
from src.tremor_analysis import tremor_score


def simulate_tremor(duration=10, sampling_rate=30, tremor_freq=5):
    t = np.linspace(0, duration, duration * sampling_rate)
    tremor = 2 * np.sin(2 * np.pi * tremor_freq * t)
    noise = np.random.normal(0, 0.3, len(t))
    signal = tremor + noise
    return t, signal


def simulate_healthy(duration=10, sampling_rate=30):
    t = np.linspace(0, duration, duration * sampling_rate)
    signal = np.random.normal(0, 0.2, len(t))
    return t, signal


SAMPLING_RATE = 30

t1, healthy = simulate_healthy()
t2, tremor = simulate_tremor()

healthy_result = tremor_score(healthy, SAMPLING_RATE)
tremor_result = tremor_score(tremor, SAMPLING_RATE)

print("Healthy Result:", healthy_result)
print("Tremor Result:", tremor_result)

plt.plot(t2, tremor)
plt.title("Tremor Signal")
plt.show()