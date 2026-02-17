from src.tremor_analysis import tremor_score
from src.session_tracker import SessionTracker
import numpy as np


def simulate_tremor(freq):
    t = np.linspace(0, 10, 300)
    tremor = 2 * np.sin(2 * np.pi * freq * t)
    noise = np.random.normal(0, 0.3, len(t))
    return tremor + noise


tracker = SessionTracker(baseline_sessions=3)

# Simulate 3 baseline sessions
for _ in range(3):
    baseline_freq = np.random.uniform(2.8, 3.2)  # small natural variation
    signal = simulate_tremor(freq=baseline_freq)
    metrics = tremor_score(signal, 30)
    tracker.add_session(metrics)

print("Baseline built.\n")

# New session (simulate deterioration)
new_signal = simulate_tremor(freq=5)
new_metrics = tremor_score(new_signal, 30)

result = tracker.evaluate_session(new_metrics)

print("New Session Evaluation:")
print(result)