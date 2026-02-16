import numpy as np


class SessionTracker:
    def __init__(self, baseline_sessions=3):
        self.baseline_sessions = baseline_sessions
        self.sessions = []
        self.baseline = None

    def add_session(self, metrics):
        """
        metrics = {
            "variance": float,
            "amplitude": float,
            "dominant_frequency": float
        }
        """
        self.sessions.append(metrics)

        if len(self.sessions) == self.baseline_sessions:
            self._compute_baseline()

    def _compute_baseline(self):
        baseline = {}

        for key in self.sessions[0].keys():
            values = [session[key] for session in self.sessions]
            std_val = np.std(values)
            if std_val < 0.01:
                std_val = 0.01

            baseline[key] = {
                "mean": float(np.mean(values)),
                "std": float(std_val)
            }

        self.baseline = baseline

    def evaluate_session(self, metrics):
        if self.baseline is None:
            return {
                "status": "Collecting baseline data",
                "sessions_needed": self.baseline_sessions - len(self.sessions)
            }

        deviations = {}

        for key in ["variance", "amplitude", "dominant_frequency"]:
            mean = self.baseline[key]["mean"]
            std = self.baseline[key]["std"]

            z_score = (metrics[key] - mean) / std

            # Cap unrealistic frequency deviation
            if key == "dominant_frequency":
                if z_score > 5:
                    z_score = 5
                elif z_score < -5:
                    z_score = -5

            deviations[key] = float(z_score)

        risk = self._risk_band(deviations)

        return {
            "deviations": deviations,
            "risk_level": risk
        }
    def _risk_band(self, deviations):
        max_dev = max(abs(v) for v in deviations.values())

        if max_dev < 1:
            return "Stable"
        elif max_dev < 2:
            return "Mild Change"
        else:
            return "Significant Change"