"""
utils/metrics.py
----------------
Evaluation utilities for FaultSense — RUL regression and anomaly detection.

RUL metrics
  rmse                   Root Mean Squared Error (cycles)
  nasa_score             Official NASA PHM asymmetric scoring function
  mean_absolute_error    Mean Absolute Error (cycles)

Anomaly detection metrics
  binary_labels            Threshold scores → 0/1 predictions
  binary_labels_persistent Threshold with min N consecutive cycles (noise filter)
  detection_lead_time      Cycles of early warning before true fault
  early_detection_rate     Fraction of engines detected >= min_lead cycles early
  false_alarm_rate         False alarms per 100 healthy cycles (absolute rate)
  precision_recall_f1      TP/FP/FN → P, R, F1

Full evaluation
  evaluate_detector        P/R/F1 + lead time across all engines (micro or macro avg)
  compare_detectors        LSTM vs EKF side-by-side with delta metrics
"""

import numpy as np
from typing import Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RUL METRICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Root Mean Squared Error between predicted and true RUL (cycles).

    Squaring amplifies large errors: an error of 40 cycles contributes
    16× more than an error of 10 cycles. Use when large errors are
    especially unacceptable. Target on FD001: < 15 cycles.

    Formula: √( (1/N) Σᵢ (ŷᵢ − yᵢ)² )
    """
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def nasa_score(y_pred: np.ndarray, y_true: np.ndarray,
               c_early: float = 13.0,
               c_late:  float = 10.0) -> float:
    """
    NASA PHM asymmetric scoring function. Lower is better, no upper bound.

    Penalises late predictions (optimistic — predicting more life than
    remains) more than early predictions (pessimistic — predicting less
    life than remains). Reflects real-world safety asymmetry: predicting
    an engine will last longer than it will is more dangerous than
    scheduling maintenance slightly too early.

    d = y_pred − y_true
      d < 0  → early (pessimistic): S = exp(−d / c_early) − 1
      d ≥ 0  → late  (optimistic):  S = exp( d / c_late)  − 1

    NASA standard: c_early=13, c_late=10 → late predictions penalised ~1.5×
    harder per unit error than early predictions.

    Parameters
    ----------
    c_early : float
        Denominator for early predictions. Larger = gentler early penalty.
    c_late : float
        Denominator for late predictions. Larger = gentler late penalty.
        Keep c_late < c_early to preserve the asymmetry (late > early).
    """
    d = y_pred - y_true
    s = np.where(d < 0,
                 np.exp(-d / c_early) - 1,
                 np.exp( d / c_late)  - 1)
    return float(s.sum())


def mean_absolute_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Mean Absolute Error (cycles). Linear penalty — all errors treated equally.
    Robust to outliers. Use alongside RMSE to detect whether errors are
    concentrated in a few engines (RMSE >> MAE) or spread evenly (RMSE ≈ MAE).

    Formula: (1/N) Σᵢ |ŷᵢ − yᵢ|
    """
    return float(np.abs(y_pred - y_true).mean())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANOMALY DETECTION — THRESHOLDING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def binary_labels(scores: np.ndarray, threshold: float) -> np.ndarray:
    """
    Convert continuous anomaly scores to binary predictions.

    ŷₜ = 1  if  scoreₜ > threshold,  else  0

    Simple point-wise threshold — a single cycle above threshold
    triggers an alarm. Use binary_labels_persistent if raw scores
    are noisy and you want to suppress single-spike false alarms.

    Parameters
    ----------
    scores    : (T,) array of anomaly scores, one per cycle
    threshold : scalar decision boundary (τ = μ + 3σ from calibration)

    Returns
    -------
    (T,) int array of 0s and 1s
    """
    return (scores > threshold).astype(int)


def binary_labels_persistent(scores: np.ndarray,
                              threshold: float,
                              min_consecutive: int = 3) -> np.ndarray:
    """
    Like binary_labels but requires the score to stay above threshold
    for at least min_consecutive consecutive cycles before declaring
    an alarm. Resets the counter if the score drops below threshold.

    This eliminates single-cycle noise spikes without changing the
    threshold itself — a softer alternative to raising τ.

    Tradeoff: adds min_consecutive cycles of detection latency in
    exchange for fewer false positives on noisy sensor readings.

    Parameters
    ----------
    scores          : (T,) array of anomaly scores
    threshold       : decision boundary
    min_consecutive : how many cycles above threshold before alarm fires

    Returns
    -------
    (T,) int array of 0s and 1s
    """
    labels = np.zeros(len(scores), dtype=int)
    count  = 0
    in_alarm = False
    for t, s in enumerate(scores):
        if s > threshold:
            count += 1
            if count >= min_consecutive:
                in_alarm = True
        else:
            count    = 0
            in_alarm = False
        labels[t] = int(in_alarm)
    return labels


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANOMALY DETECTION — TIMING METRICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detection_lead_time(scores: np.ndarray,
                         threshold: float,
                         true_fault_cycle: int) -> Optional[int]:
    """
    Cycles of early warning: how many cycles before the true fault
    onset does the detector first alarm?

      lead_time = true_fault_cycle − first_alarm_cycle

      Positive → alarm fired early  (good)
      Zero     → alarm fired exactly at fault onset
      Negative → alarm fired after fault onset  (late, dangerous)
      None     → alarm never fired  (missed fault)

    Parameters
    ----------
    scores           : (T,) anomaly score sequence for one engine
    threshold        : decision boundary
    true_fault_cycle : ground-truth cycle index where fault begins
    """
    alarm = next(
        (t for t, s in enumerate(scores) if s > threshold),
        None
    )
    if alarm is None:
        return None
    return true_fault_cycle - alarm


def early_detection_rate(score_sequences: list,
                          threshold: float,
                          fault_cycles: list,
                          min_lead: int = 10) -> float:
    """
    Fraction of engines detected at least min_lead cycles before fault.

    More operationally meaningful than pct_detected: an alarm fired
    1 cycle before failure is technically a detection but gives the
    maintenance team no time to act. early_detection_rate with
    min_lead=10 answers "how often do we give at least 10 cycles
    of useful warning?".

    Parameters
    ----------
    score_sequences : list of (Tᵢ,) arrays, one per engine
    threshold       : decision boundary
    fault_cycles    : list of true fault cycle indices
    min_lead        : minimum lead time to count as useful detection

    Returns
    -------
    float in [0, 1]
    """
    count = 0
    for scores, fc in zip(score_sequences, fault_cycles):
        lt = detection_lead_time(scores, threshold, fc)
        if lt is not None and lt >= min_lead:
            count += 1
    return round(count / len(score_sequences), 4)


def false_alarm_rate(score_sequences: list,
                     threshold: float,
                     fault_cycles: list,
                     fault_window: int = 30) -> float:
    """
    False alarms per 100 healthy cycles (absolute rate).

    Precision tells you the *ratio* of alarms that are real, but not
    how frequently false alarms occur in absolute terms. An operator
    cares about "how many unnecessary alerts do I get per week?".

    Healthy cycles are defined as cycles before [fault_cycle − fault_window].
    Only those cycles contribute to the denominator.

    Parameters
    ----------
    score_sequences : list of (Tᵢ,) arrays, one per engine
    threshold       : decision boundary
    fault_cycles    : list of true fault cycle indices
    fault_window    : cycles before fault that define the positive zone
                      (same value used in evaluate_detector)

    Returns
    -------
    float — false alarms per 100 healthy cycles
    """
    total_healthy = 0
    total_fp      = 0
    for scores, fc in zip(score_sequences, fault_cycles):
        healthy_end    = max(0, fc - fault_window)
        healthy_scores = scores[:healthy_end]
        total_fp      += int((healthy_scores > threshold).sum())
        total_healthy += len(healthy_scores)
    if total_healthy == 0:
        return 0.0
    return round((total_fp / total_healthy) * 100, 4)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANOMALY DETECTION — CLASSIFICATION METRICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def precision_recall_f1(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """
    Precision, Recall, and F1 from binary predictions and ground truth.

      TP: predicted alarm AND actually in fault zone  → correct
      FP: predicted alarm AND actually healthy        → false alarm
      FN: predicted normal AND actually in fault zone → missed fault (dangerous)
      TN: predicted normal AND actually healthy       → correct (not needed here)

      Precision = TP / (TP + FP)      of all alarms, fraction that were real
      Recall    = TP / (TP + FN)      of all real faults, fraction we caught
      F1        = 2PR / (P + R)       harmonic mean — forces both P and R high

    The 1e-10 epsilon prevents ZeroDivisionError when a detector
    never fires (TP=FP=0 → Precision would be 0/0).

    Parameters
    ----------
    y_pred : (N,) binary array — 0=normal, 1=alarm
    y_true : (N,) binary array — 0=healthy, 1=fault zone

    Returns
    -------
    dict with keys: precision, recall, f1, tp, fp, fn
    """
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp + 1e-10)
    recall    = tp / (tp + fn + 1e-10)
    f1        = 2 * precision * recall / (precision + recall + 1e-10)

    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "tp": tp, "fp": fp, "fn": fn,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FULL EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evaluate_detector(
    score_sequences: list,
    threshold: float,
    fault_cycles: list,
    fault_window: int = 30,
    average: str = "micro",
) -> dict:
    """
    Full anomaly detection evaluation across all test engines.

    Ground truth labelling
    ----------------------
    For each engine with fault at cycle fc:
      y_true[t] = 1  if  t >= max(0, fc − fault_window)
      y_true[t] = 0  otherwise
    The fault_window (default 30 cycles) defines the "positive zone".
    Cycles in that zone that go undetected count as false negatives.

    Averaging modes
    ---------------
    "micro" (default): pool all cycles from all engines into two flat
        arrays, then compute P/R/F1 once. Long engines (many cycles)
        have proportionally more weight. This is the standard in the
        CMAPSS literature.

    "macro": compute P/R/F1 per engine, then average across engines.
        Each engine has equal weight regardless of length. More
        appropriate when engine lengths vary widely and you don't want
        long engines to dominate.

    Parameters
    ----------
    score_sequences : list of (Tᵢ,) score arrays, one per engine
    threshold       : decision boundary (scalar)
    fault_cycles    : list of true fault cycle indices, one per engine
    fault_window    : cycles before fault that are labelled positive
    average         : "micro" or "macro"

    Returns
    -------
    dict with keys: precision, recall, f1, tp (micro only), fp (micro only),
                    fn (micro only), mean_lead_time, std_lead_time, pct_detected
    """
    if average == "macro":
        precisions, recalls, f1s = [], [], []
        lead_times = []
        for scores, fc in zip(score_sequences, fault_cycles):
            T      = len(scores)
            y_true = np.zeros(T, dtype=int)
            y_true[max(0, fc - fault_window):] = 1
            y_pred = binary_labels(scores, threshold)
            prf    = precision_recall_f1(y_pred, y_true)
            precisions.append(prf["precision"])
            recalls.append(prf["recall"])
            f1s.append(prf["f1"])
            lt = detection_lead_time(scores, threshold, fc)
            if lt is not None:
                lead_times.append(lt)
        return {
            "precision":      round(float(np.mean(precisions)), 4),
            "recall":         round(float(np.mean(recalls)),    4),
            "f1":             round(float(np.mean(f1s)),        4),
            "mean_lead_time": float(np.mean(lead_times)) if lead_times else None,
            "std_lead_time":  float(np.std(lead_times))  if lead_times else None,
            "pct_detected":   round(len(lead_times) / len(score_sequences), 4),
            "average":        "macro",
        }

    # ── micro (default) ───────────────────────────────────────────────────────
    all_pred, all_true = [], []
    lead_times         = []

    for scores, fc in zip(score_sequences, fault_cycles):
        T      = len(scores)
        y_true = np.zeros(T, dtype=int)
        y_true[max(0, fc - fault_window):] = 1

        y_pred = binary_labels(scores, threshold)
        all_pred.extend(y_pred.tolist())
        all_true.extend(y_true.tolist())

        lt = detection_lead_time(scores, threshold, fc)
        if lt is not None:
            lead_times.append(lt)

    prf = precision_recall_f1(np.array(all_pred), np.array(all_true))
    prf["mean_lead_time"] = float(np.mean(lead_times)) if lead_times else None
    prf["std_lead_time"]  = float(np.std(lead_times))  if lead_times else None
    prf["pct_detected"]   = round(len(lead_times) / len(score_sequences), 4)
    prf["average"]        = "micro"
    return prf


def compare_detectors(lstm_scores_list: list,
                       ekf_scores_list:  list,
                       fault_cycles:     list,
                       threshold_lstm:   float,
                       threshold_ekf:    float,
                       fault_window:     int = 30,
                       average:          str = "micro",
                       min_lead:         int = 10) -> dict:
    """
    Side-by-side evaluation of LSTM vs EKF detector.

    Each detector gets its own threshold (calibrated separately from its
    own score distribution — sharing one threshold would be unfair since
    LSTM reconstruction errors and EKF innovation residuals live on
    completely different scales).

    Parameters
    ----------
    lstm_scores_list : list of (Tᵢ,) LSTM anomaly score arrays
    ekf_scores_list  : list of (Tᵢ,) EKF anomaly score arrays
    fault_cycles     : list of true fault cycle indices
    threshold_lstm   : LSTM decision boundary
    threshold_ekf    : EKF decision boundary
    fault_window     : cycles before fault labelled as positive zone
    average          : "micro" or "macro" passed to evaluate_detector
    min_lead         : minimum lead time for early_detection_rate

    Returns
    -------
    dict with keys:
      "lstm"            : full metrics dict for LSTM
      "ekf"             : full metrics dict for EKF
      "delta_f1"        : lstm_f1 − ekf_f1  (positive = LSTM wins)
      "delta_lead_time" : lstm_lead − ekf_lead (positive = LSTM earlier)
      "delta_edr"       : lstm_edr − ekf_edr  (early detection rate diff)
      "delta_far"       : lstm_far − ekf_far  (false alarm rate diff, negative = LSTM fewer)
    """
    lstm_metrics = evaluate_detector(
        lstm_scores_list, threshold_lstm, fault_cycles, fault_window, average)
    ekf_metrics  = evaluate_detector(
        ekf_scores_list,  threshold_ekf,  fault_cycles, fault_window, average)

    lstm_edr = early_detection_rate(
        lstm_scores_list, threshold_lstm, fault_cycles, min_lead)
    ekf_edr  = early_detection_rate(
        ekf_scores_list,  threshold_ekf,  fault_cycles, min_lead)

    lstm_far = false_alarm_rate(
        lstm_scores_list, threshold_lstm, fault_cycles, fault_window)
    ekf_far  = false_alarm_rate(
        ekf_scores_list,  threshold_ekf,  fault_cycles, fault_window)

    return {
        "lstm": lstm_metrics,
        "ekf":  ekf_metrics,
        "delta_f1": round(
            lstm_metrics["f1"] - ekf_metrics["f1"], 4),
        "delta_lead_time": round(
            (lstm_metrics["mean_lead_time"] or 0) -
            (ekf_metrics["mean_lead_time"]  or 0), 1),
        "delta_edr": round(lstm_edr - ekf_edr, 4),
        "delta_far": round(lstm_far - ekf_far, 4),
        "lstm_edr":  lstm_edr,
        "ekf_edr":   ekf_edr,
        "lstm_far":  lstm_far,
        "ekf_far":   ekf_far,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# QUICK SANITY CHECK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    np.random.seed(42)
    n  = 200
    fc = 140
    thr = 0.50

    # Synthetic score stream: flat noise then rising degradation
    scores = np.concatenate([
        np.random.uniform(0.05, 0.15, 100),
        np.linspace(0.15, 0.95, 100) + np.random.randn(100) * 0.04,
    ])

    print("── Detection metrics ──────────────────────────────")
    lt = detection_lead_time(scores, thr, fc)
    print(f"Lead time (simple):     {lt} cycles")

    labels_simple     = binary_labels(scores, thr)
    labels_persistent = binary_labels_persistent(scores, thr, min_consecutive=3)
    print(f"Alarms (simple):        {labels_simple.sum()} cycles")
    print(f"Alarms (persistent 3):  {labels_persistent.sum()} cycles")

    y_true = np.zeros(n, dtype=int)
    y_true[fc - 30:] = 1
    prf = precision_recall_f1(labels_simple, y_true)
    print(f"Precision: {prf['precision']:.3f}  "
          f"Recall: {prf['recall']:.3f}  "
          f"F1: {prf['f1']:.3f}")

    edr = early_detection_rate([scores], thr, [fc], min_lead=10)
    far = false_alarm_rate([scores], thr, [fc], fault_window=30)
    print(f"Early detection rate (≥10 cyc): {edr:.2%}")
    print(f"False alarm rate per 100 cyc:   {far:.2f}")

    print("\n── RUL metrics ────────────────────────────────────")
    y_pred_rul = np.array([180.0, 45.0, 92.0])
    y_true_rul = np.array([175.0, 50.0, 88.0])
    print(f"RMSE:      {rmse(y_pred_rul, y_true_rul):.2f} cycles")
    print(f"MAE:       {mean_absolute_error(y_pred_rul, y_true_rul):.2f} cycles")
    print(f"NASA score (standard):    {nasa_score(y_pred_rul, y_true_rul):.4f}")
    print(f"NASA score (symmetric):   "
          f"{nasa_score(y_pred_rul, y_true_rul, c_early=10, c_late=10):.4f}")
