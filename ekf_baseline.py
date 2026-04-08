"""
models/ekf_baseline.py
-----------------------
Extended Kalman Filter baseline for anomaly detection.

Models each sensor independently as a slowly-drifting process:
  State:    x = [value, drift_rate]
  Dynamics: x_{t+1} = F*x_t + w,    w ~ N(0, Q)
  Measure:  y_t     = H*x_t + v,    v ~ N(0, R)

Anomaly score = normalized Mahalanobis distance of the innovation
  score = (y - H*x̂)^T * S^{-1} * (y - H*x̂)
where S = H*P*H^T + R is the innovation covariance.

This is the classical approach. We run it per sensor and aggregate.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class EKFSensorState:
    """KF state for a single sensor."""
    x: np.ndarray       # state [value, drift_rate]   shape (2,)
    P: np.ndarray       # covariance                  shape (2, 2)
    Q: np.ndarray       # process noise               shape (2, 2)
    R: float            # measurement noise variance
    history: list = field(default_factory=list)   # innovation history


class EKFBaseline:
    """
    Per-sensor Kalman filter anomaly detector.

    Usage
    -----
    ekf = EKFBaseline(n_sensors=14)
    ekf.initialize(X_init)           # fit Q, R from first N cycles
    score = ekf.update_and_score(y)  # step-by-step

    Or batch:
    scores = ekf.run(X)              # shape (T,)
    """

    def __init__(
        self,
        n_sensors:    int   = 14,
        process_noise: float = 1e-4,   # Q diagonal baseline — tune per dataset
        meas_noise:    float = 1e-2,   # R diagonal baseline
        ema_alpha:     float = 0.3,    # exponential smoothing on final score
    ):
        self.n_sensors     = n_sensors
        self.process_noise = process_noise
        self.meas_noise    = meas_noise
        self.ema_alpha     = ema_alpha
        self.states: list[EKFSensorState] = []
        self.threshold: Optional[float]   = None
        self._ema_score: float            = 0.0

        # State transition: x_{t+1} = F*x_t
        # [value_{t+1}]   [1  dt] [value_t     ]
        # [drift_{t+1} ] = [0   1] [drift_rate_t]
        self.F = np.array([[1.0, 1.0],
                           [0.0, 1.0]])

        # Observation: y = H*x   (we observe only the value, not drift)
        self.H = np.array([[1.0, 0.0]])

    # ── Initialisation ──────────────────────────────────────────────────────

    def initialize(self, X_init: np.ndarray, tune_noise: bool = True):
        """
        Initialize filter states from observed data.
        X_init : (T, n_sensors) — use first ~50 cycles per engine
        tune_noise : estimate Q and R from init data variance
        """
        self.states = []
        for s in range(self.n_sensors):
            vals = X_init[:, s]

            # Estimate measurement noise R from short-window variance
            R = float(np.var(np.diff(vals)) / 2 + 1e-6) if tune_noise \
                else self.meas_noise

            # Estimate process noise Q from drift rate variance
            drift_var = float(np.var(np.diff(vals, n=2)) / 2 + 1e-8) if tune_noise \
                else self.process_noise

            Q = np.array([[drift_var * 0.1, 0.0],
                          [0.0,              drift_var]])

            # Initial state: start at first observation
            x0 = np.array([vals[0], 0.0])
            P0 = np.eye(2) * R * 10   # start uncertain

            self.states.append(EKFSensorState(x=x0, P=P0, Q=Q, R=R))

    def reset(self):
        """Reset all filter states to initial conditions."""
        for s in self.states:
            s.x       = np.array([0.0, 0.0])
            s.P       = np.eye(2) * 0.1
            s.history = []
        self._ema_score = 0.0

    # ── Single-step update ──────────────────────────────────────────────────

    def _step_sensor(self, state: EKFSensorState, y: float) -> float:
        """
        One KF predict-update cycle for a single sensor.
        Returns normalized innovation score (Mahalanobis-like).
        """
        F, H = self.F, self.H

        # Predict
        x_pred = F @ state.x
        P_pred = F @ state.P @ F.T + state.Q

        # Innovation
        innov  = y - (H @ x_pred)[0]           # scalar
        S      = (H @ P_pred @ H.T)[0, 0] + state.R  # innovation variance
        norm_innov = innov ** 2 / (S + 1e-10)  # χ²(1) under nominal

        # Kalman gain
        K = (P_pred @ H.T) / S                 # (2, 1) then squeeze
        K = K.flatten()

        # Update
        state.x = x_pred + K * innov
        state.P = (np.eye(2) - np.outer(K, H)) @ P_pred

        state.history.append(float(norm_innov))
        return float(norm_innov)

    def update_and_score(self, y: np.ndarray) -> dict:
        """
        Feed one timestep of sensor readings and return anomaly info.
        y : (n_sensors,)
        """
        assert len(self.states) == self.n_sensors, "Call initialize() first."

        per_sensor_scores = np.array([
            self._step_sensor(self.states[s], y[s])
            for s in range(self.n_sensors)
        ])

        raw_score = float(per_sensor_scores.mean())

        # Exponential smoothing
        self._ema_score = (self.ema_alpha * raw_score
                           + (1 - self.ema_alpha) * self._ema_score)

        return {
            "raw_score":       raw_score,
            "ema_score":       self._ema_score,
            "per_sensor":      per_sensor_scores,
            "top_sensor":      int(per_sensor_scores.argmax()),
            "is_anomaly": (self._ema_score > self.threshold)
                           if self.threshold is not None else None,
        }

    # ── Batch processing ────────────────────────────────────────────────────

    def run(self, X: np.ndarray) -> np.ndarray:
        """
        Run EKF over a full sensor trajectory.
        X : (T, n_sensors)
        Returns : (T,) EMA anomaly scores
        """
        scores = []
        self._ema_score = 0.0
        for t in range(len(X)):
            result = self.update_and_score(X[t])
            scores.append(result["ema_score"])
        return np.array(scores)

    # ── Threshold calibration ────────────────────────────────────────────────

    def calibrate_threshold(self, X_healthy: np.ndarray,
                             k_sigma: float = 3.0) -> float:
        """
        Calibrate anomaly threshold from healthy engine data.
        X_healthy : (T_total, n_sensors) — concatenation of healthy cycles
        """
        self.reset()
        scores = self.run(X_healthy)
        mu     = scores.mean()
        sig    = scores.std()
        self.threshold = float(mu + k_sigma * sig)
        print(f"EKF threshold calibrated: μ={mu:.4f}  σ={sig:.4f}  "
              f"threshold={self.threshold:.4f}")
        return self.threshold

    # ── Multi-engine evaluation ──────────────────────────────────────────────

    def evaluate_engines(self, X_engines: list[np.ndarray],
                          true_fault_cycles: Optional[list[int]] = None):
        """
        Evaluate detection lead time across multiple engines.
        X_engines : list of (T_i, n_sensors) arrays
        true_fault_cycles : cycle index where fault is known to begin
        Returns summary dict with precision, recall, lead_time stats.
        """
        detections = []
        for i, X in enumerate(X_engines):
            self.reset()
            scores = self.run(X)
            detected = next(
                (t for t, s in enumerate(scores) if s > self.threshold), None)
            detections.append(detected)

        detected_arr = np.array([d for d in detections if d is not None])

        result = {"n_engines": len(X_engines),
                  "n_detected": len(detected_arr)}

        if true_fault_cycles is not None:
            tc = np.array(true_fault_cycles)
            lead_times = tc - detected_arr[:len(tc)]
            result["mean_lead_time"] = float(lead_times.mean())
            result["std_lead_time"]  = float(lead_times.std())

        return result


if __name__ == "__main__":
    # Smoke test with synthetic data
    np.random.seed(42)
    T, S = 200, 14
    X_healthy = np.random.randn(100, S) * 0.1

    # Degraded: gradual drift after cycle 120
    X_degraded = np.vstack([
        np.random.randn(120, S) * 0.1,
        np.random.randn(80,  S) * 0.1 + np.linspace(0, 2, 80)[:, None]
    ])

    ekf = EKFBaseline(n_sensors=S)
    ekf.initialize(X_healthy)
    ekf.calibrate_threshold(X_healthy)

    print("\nRunning on degraded engine …")
    scores = ekf.run(X_degraded)
    first_alarm = next((i for i, s in enumerate(scores) if s > ekf.threshold), None)
    print(f"First alarm at cycle: {first_alarm}  (fault starts at cycle 120)")
    print(f"Detection lead time:  {120 - first_alarm if first_alarm else 'missed'} cycles")
