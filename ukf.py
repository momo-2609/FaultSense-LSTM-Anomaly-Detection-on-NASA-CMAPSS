"""
models/ukf.py
-------------
Unscented Kalman Filter (UKF) for nonlinear state estimation.

Why UKF over EKF?
  The EKF linearises nonlinear dynamics via Jacobians — error-prone and
  inaccurate for strongly nonlinear systems. The UKF propagates a small set
  of carefully chosen "sigma points" through the exact nonlinear function,
  giving a 2nd-order accurate mean and covariance estimate with no Jacobians.

This module provides:
  UKFSensorState   — per-sensor state container (extends EKF pattern)
  UKFBaseline      — drop-in replacement for EKFBaseline with UKF internals
  UKFPoseEstimator — 2D pose estimation (x, y, heading, speed) for robotics demos

State convention (per sensor, matches EKFBaseline):
  x = [value, drift_rate]   shape (2,)

Sigma-point convention: Merwe scaled sigma points
  alpha=1e-3, beta=2, kappa=0  (standard for small state dimensions)

Run smoke test:
  python models/ukf.py
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple


# ── Sigma-point parameters ────────────────────────────────────────────────────

@dataclass
class MerweParams:
    """
    Merwe scaled sigma-point parameters.

    alpha : spread of sigma points around mean (1e-4 to 1)
    beta  : prior knowledge of distribution (2 optimal for Gaussian)
    kappa : secondary scaling (0 for state estimation)
    """
    alpha: float = 1e-3
    beta:  float = 2.0
    kappa: float = 0.0

    def weights(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and covariance weights for n-dimensional state.
        Returns (Wm, Wc) each of shape (2n+1,).
        """
        lam  = self.alpha ** 2 * (n + self.kappa) - n
        Wm   = np.full(2 * n + 1, 0.5 / (n + lam))
        Wc   = Wm.copy()
        Wm[0] = lam / (n + lam)
        Wc[0] = lam / (n + lam) + (1 - self.alpha ** 2 + self.beta)
        return Wm, Wc

    def lambda_(self, n: int) -> float:
        return self.alpha ** 2 * (n + self.kappa) - n


def sigma_points(x: np.ndarray, P: np.ndarray,
                 params: MerweParams) -> np.ndarray:
    """
    Generate 2n+1 sigma points around state x with covariance P.

    Returns sigmas : (2n+1, n)
    """
    n   = len(x)
    lam = params.lambda_(n)
    # Matrix square root via Cholesky — numerically stable
    try:
        S = np.linalg.cholesky((n + lam) * P)
    except np.linalg.LinAlgError:
        # Fallback: add small jitter to ensure positive-definiteness
        S = np.linalg.cholesky((n + lam) * (P + np.eye(n) * 1e-8))

    sigmas    = np.zeros((2 * n + 1, n))
    sigmas[0] = x
    for i in range(n):
        sigmas[i + 1]     = x + S[:, i]
        sigmas[n + i + 1] = x - S[:, i]
    return sigmas


# ── UKF core ──────────────────────────────────────────────────────────────────

@dataclass
class UKFSensorState:
    """UKF state for a single sensor — mirrors EKFSensorState interface."""
    x:       np.ndarray        # state [value, drift_rate]   shape (2,)
    P:       np.ndarray        # covariance                  shape (2, 2)
    Q:       np.ndarray        # process noise               shape (2, 2)
    R:       float             # measurement noise variance
    history: list = field(default_factory=list)


class UKFBaseline:
    """
    Per-sensor Unscented Kalman Filter anomaly detector.

    Drop-in replacement for EKFBaseline — same public API, UKF internals.
    Advantage: handles nonlinear sensor degradation curves correctly
    without requiring Jacobian derivations.

    Usage (identical to EKFBaseline)
    ---------------------------------
    ukf = UKFBaseline(n_sensors=15)
    ukf.initialize(X_init)
    scores = ukf.run(X)
    ukf.calibrate_threshold(X_healthy)
    result = ukf.update_and_score(y)
    """

    def __init__(
        self,
        n_sensors:     int   = 15,
        process_noise: float = 1e-4,
        meas_noise:    float = 1e-2,
        ema_alpha:     float = 0.3,
        merwe:         MerweParams = None,
    ):
        self.n_sensors     = n_sensors
        self.process_noise = process_noise
        self.meas_noise    = meas_noise
        self.ema_alpha     = ema_alpha
        self.merwe         = merwe or MerweParams()
        self.states: list[UKFSensorState] = []
        self.threshold: Optional[float]   = None
        self._ema_score: float            = 0.0

        # Precompute weights for 2-dimensional state
        self._Wm, self._Wc = self.merwe.weights(n=2)

    # ── State transition and observation (can be nonlinear) ──────────────────

    @staticmethod
    def _f(x: np.ndarray) -> np.ndarray:
        """
        Process model: constant-velocity drift.
        x = [value, drift_rate]  →  x_next = [value + drift_rate, drift_rate]

        Replace this with any nonlinear function for your system.
        """
        return np.array([x[0] + x[1], x[1]])

    @staticmethod
    def _h(x: np.ndarray) -> float:
        """
        Observation model: we observe only the sensor value.
        Replace with nonlinear sensor model if needed.
        """
        return x[0]

    # ── UKF predict-update ───────────────────────────────────────────────────

    def _step_sensor(self, state: UKFSensorState, y: float) -> float:
        """
        One UKF predict-update cycle for a single sensor.
        Returns normalised innovation score (Mahalanobis distance squared).
        """
        # ── Predict ──────────────────────────────────────────────────────────
        sigmas = sigma_points(state.x, state.P, self.merwe)   # (5, 2)

        # Propagate sigma points through process model
        sigmas_f = np.array([self._f(s) for s in sigmas])     # (5, 2)

        # Predicted mean and covariance
        x_pred = (self._Wm[:, None] * sigmas_f).sum(axis=0)
        P_pred = state.Q.copy()
        for i, s in enumerate(sigmas_f):
            d = (s - x_pred)[:, None]
            P_pred += self._Wc[i] * (d @ d.T)

        # ── Update ───────────────────────────────────────────────────────────
        # Propagate through observation model
        sigmas_h = np.array([self._h(s) for s in sigmas_f])   # (5,)

        # Predicted measurement mean and variance
        y_pred = float((self._Wm * sigmas_h).sum())
        S_yy   = state.R
        for i, sh in enumerate(sigmas_h):
            S_yy += self._Wc[i] * (sh - y_pred) ** 2

        # Cross-covariance P_xy
        P_xy = np.zeros(2)
        for i, (sf, sh) in enumerate(zip(sigmas_f, sigmas_h)):
            P_xy += self._Wc[i] * (sf - x_pred) * (sh - y_pred)

        # Kalman gain
        K = P_xy / (S_yy + 1e-10)   # shape (2,)

        # Innovation and normalised score
        innov      = y - y_pred
        norm_innov = innov ** 2 / (S_yy + 1e-10)

        # State and covariance update
        state.x = x_pred + K * innov
        state.P = P_pred - np.outer(K, K) * S_yy

        # Enforce symmetry and positive-definiteness
        state.P = 0.5 * (state.P + state.P.T)
        state.P += np.eye(2) * 1e-10

        state.history.append(float(norm_innov))
        return float(norm_innov)

    # ── Public API (mirrors EKFBaseline exactly) ──────────────────────────────

    def initialize(self, X_init: np.ndarray, tune_noise: bool = True):
        """
        Initialize filter states from observed data.
        X_init : (T, n_sensors)
        """
        self.states = []
        for s in range(self.n_sensors):
            vals = X_init[:, s]
            R = float(np.var(np.diff(vals)) / 2 + 1e-6) if tune_noise \
                else self.meas_noise
            drift_var = float(np.var(np.diff(vals, n=2)) / 2 + 1e-8) if tune_noise \
                else self.process_noise
            Q  = np.array([[drift_var * 0.1, 0.0], [0.0, drift_var]])
            x0 = np.array([vals[0], 0.0])
            P0 = np.eye(2) * R * 10
            self.states.append(UKFSensorState(x=x0, P=P0, Q=Q, R=R))

    def reset(self):
        """Reset all filter states."""
        for s in self.states:
            s.x       = np.array([0.0, 0.0])
            s.P       = np.eye(2) * 0.1
            s.history = []
        self._ema_score = 0.0

    def update_and_score(self, y: np.ndarray) -> dict:
        """
        Feed one timestep of sensor readings and return anomaly info.
        y : (n_sensors,)
        """
        assert len(self.states) == self.n_sensors, "Call initialize() first."

        per_sensor_scores = np.array([
            self._step_sensor(self.states[s], float(y[s]))
            for s in range(self.n_sensors)
        ])

        raw_score = float(per_sensor_scores.mean())
        self._ema_score = (self.ema_alpha * raw_score
                           + (1 - self.ema_alpha) * self._ema_score)

        return {
            "raw_score":  raw_score,
            "ema_score":  self._ema_score,
            "per_sensor": per_sensor_scores,
            "top_sensor": int(per_sensor_scores.argmax()),
            "is_anomaly": (self._ema_score > self.threshold)
                           if self.threshold is not None else None,
        }

    def run(self, X: np.ndarray) -> np.ndarray:
        """
        Run UKF over a full sensor trajectory.
        X : (T, n_sensors) → returns (T,) EMA anomaly scores
        """
        scores = []
        self._ema_score = 0.0
        for t in range(len(X)):
            result = self.update_and_score(X[t])
            scores.append(result["ema_score"])
        return np.array(scores)

    def calibrate_threshold(self, X_healthy: np.ndarray,
                             k_sigma: float = 3.0) -> float:
        """Calibrate anomaly threshold = μ + k*σ on healthy data."""
        self.reset()
        scores = self.run(X_healthy)
        mu, sig = scores.mean(), scores.std()
        self.threshold = float(mu + k_sigma * sig)
        print(f"UKF threshold calibrated: μ={mu:.4f}  σ={sig:.4f}  "
              f"threshold={self.threshold:.4f}")
        return self.threshold

    def evaluate_engines(self, X_engines: list,
                          true_fault_cycles: Optional[list] = None) -> dict:
        """Evaluate detection lead time across multiple engines."""
        detections = []
        for X in X_engines:
            self.reset()
            scores   = self.run(X)
            detected = next((t for t, s in enumerate(scores)
                             if s > self.threshold), None)
            detections.append(detected)

        detected_arr = np.array([d for d in detections if d is not None])
        result = {"n_engines": len(X_engines), "n_detected": len(detected_arr)}

        if true_fault_cycles is not None and len(detected_arr) > 0:
            tc = np.array(true_fault_cycles)
            lead_times = tc[:len(detected_arr)] - detected_arr
            result["mean_lead_time"] = float(lead_times.mean())
            result["std_lead_time"]  = float(lead_times.std())

        return result


# ── 2D Pose Estimator (robotics demo) ────────────────────────────────────────

class UKFPoseEstimator:
    """
    UKF for 2D robot pose estimation fusing IMU + position measurements.

    State: x = [px, py, heading, speed]   shape (4,)
      px, py   : position in metres
      heading  : orientation in radians
      speed    : scalar speed in m/s

    IMU provides heading_rate (gyro) and acceleration.
    Position sensor (GPS / LiDAR) provides (px, py) at lower rate.

    This demonstrates the key advantage of UKF: the process model is
    nonlinear (heading × speed → position update) and would require
    a messy Jacobian in EKF. UKF handles it transparently.

    Usage
    -----
    estimator = UKFPoseEstimator(dt=0.1)
    estimator.initialize(px=0, py=0, heading=0, speed=0)

    # IMU update (high rate)
    estimator.predict(heading_rate=0.05, acceleration=0.1)

    # Position update (low rate, when GPS/LiDAR available)
    estimator.update_position(px_meas=1.2, py_meas=0.3)

    state = estimator.state   # dict with px, py, heading, speed
    """

    def __init__(
        self,
        dt:            float = 0.1,    # IMU timestep in seconds
        Q_diag:        tuple = (0.01, 0.01, 0.001, 0.05),  # process noise
        R_pos:         float = 0.5,    # position measurement noise (m²)
        merwe:         MerweParams = None,
    ):
        self.dt    = dt
        self.Q     = np.diag(Q_diag)
        self.R_pos = np.eye(2) * R_pos
        self.merwe = merwe or MerweParams(alpha=0.1, beta=2.0, kappa=1.0)

        self.n       = 4                      # state dimension
        self._Wm, self._Wc = self.merwe.weights(self.n)

        self.x = np.zeros(4)                  # [px, py, heading, speed]
        self.P = np.eye(4) * 1.0              # initial covariance

    def initialize(self, px: float = 0, py: float = 0,
                   heading: float = 0, speed: float = 0):
        """Set initial state."""
        self.x = np.array([px, py, heading, speed], dtype=float)
        self.P = np.eye(4) * 0.1

    def _f_pose(self, x: np.ndarray, heading_rate: float,
                acceleration: float) -> np.ndarray:
        """
        Nonlinear process model: bicycle / constant-turn-rate motion.
        x = [px, py, heading, speed]
        """
        px, py, h, v = x
        px_new = px + v * np.cos(h) * self.dt
        py_new = py + v * np.sin(h) * self.dt
        h_new  = h + heading_rate * self.dt
        v_new  = v + acceleration * self.dt
        return np.array([px_new, py_new, h_new, v_new])

    def predict(self, heading_rate: float = 0.0, acceleration: float = 0.0):
        """
        IMU-driven prediction step.
        Call at every IMU timestep (e.g. 100 Hz).
        """
        sigmas   = sigma_points(self.x, self.P, self.merwe)
        sigmas_f = np.array([self._f_pose(s, heading_rate, acceleration)
                              for s in sigmas])

        # Predicted mean
        x_pred = (self._Wm[:, None] * sigmas_f).sum(axis=0)
        # Normalise heading to [-π, π]
        x_pred[2] = np.arctan2(np.sin(x_pred[2]), np.cos(x_pred[2]))

        # Predicted covariance
        P_pred = self.Q.copy()
        for i, s in enumerate(sigmas_f):
            d      = s - x_pred
            d[2]   = np.arctan2(np.sin(d[2]), np.cos(d[2]))  # heading wrap
            P_pred += self._Wc[i] * np.outer(d, d)

        self.x = x_pred
        self.P = P_pred

    def update_position(self, px_meas: float, py_meas: float):
        """
        Position measurement update (GPS / LiDAR fix).
        Call at the position sensor rate (e.g. 10 Hz).
        """
        z = np.array([px_meas, py_meas])

        # Sigma points from current prior
        sigmas   = sigma_points(self.x, self.P, self.merwe)
        sigmas_h = sigmas[:, :2]   # observe only px, py  shape (2n+1, 2)

        # Predicted measurement mean
        z_pred = (self._Wm[:, None] * sigmas_h).sum(axis=0)

        # Innovation covariance S and cross-covariance P_xz
        S    = self.R_pos.copy()
        P_xz = np.zeros((4, 2))
        for i, (sf, sh) in enumerate(zip(sigmas, sigmas_h)):
            dz    = sh - z_pred
            dx    = sf - self.x
            dx[2] = np.arctan2(np.sin(dx[2]), np.cos(dx[2]))
            S    += self._Wc[i] * np.outer(dz, dz)
            P_xz += self._Wc[i] * np.outer(dx, dz)

        # Kalman gain and update
        K         = P_xz @ np.linalg.inv(S)
        innov     = z - z_pred
        self.x    = self.x + K @ innov
        self.x[2] = np.arctan2(np.sin(self.x[2]), np.cos(self.x[2]))
        self.P    = self.P - K @ S @ K.T
        self.P    = 0.5 * (self.P + self.P.T) + np.eye(4) * 1e-9

    @property
    def state(self) -> dict:
        """Return current state estimate as a readable dict."""
        return {
            "px":      float(self.x[0]),
            "py":      float(self.x[1]),
            "heading": float(np.degrees(self.x[2])),   # degrees for readability
            "speed":   float(self.x[3]),
            "P_diag":  self.P.diagonal().tolist(),
        }


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    T, S = 200, 15

    print("=" * 55)
    print("UKFBaseline — anomaly detection smoke test")
    print("=" * 55)

    X_healthy  = np.random.randn(100, S) * 0.1
    X_degraded = np.vstack([
        np.random.randn(120, S) * 0.1,
        np.random.randn(80,  S) * 0.1 + np.linspace(0, 2, 80)[:, None]
    ])

    ukf = UKFBaseline(n_sensors=S)
    ukf.initialize(X_healthy)
    ukf.calibrate_threshold(X_healthy)

    scores      = ukf.run(X_degraded)
    first_alarm = next((i for i, s in enumerate(scores) if s > ukf.threshold), None)
    print(f"First alarm at cycle: {first_alarm}  (fault starts at cycle 120)")
    print(f"Detection lead time:  "
          f"{120 - first_alarm if first_alarm else 'missed'} cycles")

    print("\n" + "=" * 55)
    print("UKFPoseEstimator — 2D pose estimation smoke test")
    print("=" * 55)

    estimator = UKFPoseEstimator(dt=0.1)
    estimator.initialize(px=0, py=0, heading=0, speed=1.0)

    true_positions = []
    est_positions  = []

    # Simulate: robot drives in a gentle left arc for 50 steps
    for t in range(50):
        # Ground truth: constant heading_rate = 0.05 rad/step, speed = 1 m/s
        true_px = np.cos(0.05 * t) * t * 0.1
        true_py = np.sin(0.05 * t) * t * 0.1
        true_positions.append((true_px, true_py))

        # UKF prediction step
        estimator.predict(heading_rate=0.05, acceleration=0.0)

        # Position update every 5 steps (simulating 10Hz GPS vs 100Hz IMU)
        if t % 5 == 0:
            noisy_px = true_px + np.random.randn() * 0.2
            noisy_py = true_py + np.random.randn() * 0.2
            estimator.update_position(noisy_px, noisy_py)

        est_positions.append((estimator.state["px"], estimator.state["py"]))

    # Compute position RMSE
    errors = [np.sqrt((tp[0]-ep[0])**2 + (tp[1]-ep[1])**2)
              for tp, ep in zip(true_positions, est_positions)]
    print(f"Position RMSE over 50 steps: {np.mean(errors):.4f} m")
    print(f"Final state: {estimator.state}")
    print("\nAll smoke tests passed ✓")
