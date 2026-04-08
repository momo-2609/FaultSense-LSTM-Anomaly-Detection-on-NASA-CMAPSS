"""
utils/metrics.py
----------------
Evaluation utilities: RMSE, NASA scoring function,
detection lead time, precision/recall for anomaly detection.
"""

import numpy as np
from typing import Optional


# ── RUL metrics ──────────────────────────────────────────────────────────────

def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def nasa_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    NASA prognostics scoring function.
    Penalises late predictions (optimistic) more than early ones.
    Lower = better.

    S = Σ exp(-d/13) - 1   if d < 0  (late prediction)
      = Σ exp(d/10)  - 1   if d ≥ 0  (early prediction)

    where d = y_pred - y_true
    """
    d = y_pred - y_true
    s = np.where(d < 0,
                 np.exp(-d / 13) - 1,
                 np.exp(d / 10)  - 1)
    return float(s.sum())


def mean_absolute_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.abs(y_pred - y_true).mean())


# ── Anomaly detection metrics ─────────────────────────────────────────────────

def detection_lead_time(scores: np.ndarray, threshold: float,
                         true_fault_cycle: int) -> Optional[int]:
    """
    Number of cycles before the true fault that the detector first alarms.
    Positive = early detection. Negative = missed/late.
    None = never detected.
    """
    alarm = next((t for t, s in enumerate(scores) if s > threshold), None)
    if alarm is None:
        return None
    return true_fault_cycle - alarm


def binary_labels(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores > threshold).astype(int)


def precision_recall_f1(y_pred: np.ndarray, y_true: np.ndarray):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp + 1e-10)
    recall    = tp / (tp + fn + 1e-10)
    f1        = 2 * precision * recall / (precision + recall + 1e-10)

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }


def evaluate_detector(
    score_sequences: list[np.ndarray],   # one per engine
    thresholds_or_scalar,
    fault_cycles: list[int],
    fault_window: int = 30,               # cycles before fault = positive zone
) -> dict:
    """
    Full anomaly detection evaluation across multiple engines.

    Labels ground truth as:
      y=1 for cycles in [fault_cycle - fault_window, end]
      y=0 otherwise

    Returns precision, recall, F1, mean lead time.
    """
    all_pred, all_true = [], []
    lead_times         = []

    thr = thresholds_or_scalar

    for scores, fc in zip(score_sequences, fault_cycles):
        T = len(scores)
        y_true = np.zeros(T, dtype=int)
        y_true[max(0, fc - fault_window):] = 1

        y_pred = binary_labels(scores, thr)
        all_pred.extend(y_pred.tolist())
        all_true.extend(y_true.tolist())

        lt = detection_lead_time(scores, thr, fc)
        if lt is not None:
            lead_times.append(lt)

    prf = precision_recall_f1(
        np.array(all_pred),
        np.array(all_true),
    )
    prf["mean_lead_time"] = float(np.mean(lead_times)) if lead_times else None
    prf["std_lead_time"]  = float(np.std(lead_times))  if lead_times else None
    prf["pct_detected"]   = round(len(lead_times) / len(score_sequences), 4)

    return prf


# ── Comparison utility ────────────────────────────────────────────────────────

def compare_detectors(lstm_scores_list, ekf_scores_list,
                       fault_cycles, threshold_lstm, threshold_ekf,
                       fault_window=30) -> dict:
    """Side-by-side evaluation of LSTM vs EKF."""
    lstm_metrics = evaluate_detector(lstm_scores_list, threshold_lstm,
                                      fault_cycles, fault_window)
    ekf_metrics  = evaluate_detector(ekf_scores_list, threshold_ekf,
                                      fault_cycles, fault_window)

    return {
        "lstm": lstm_metrics,
        "ekf":  ekf_metrics,
        "delta_f1":        round(lstm_metrics["f1"] - ekf_metrics["f1"], 4),
        "delta_lead_time": round(
            (lstm_metrics["mean_lead_time"] or 0) -
            (ekf_metrics["mean_lead_time"]  or 0), 1
        ),
    }


if __name__ == "__main__":
    # Quick sanity check
    np.random.seed(42)
    n = 200
    scores = np.linspace(0, 1, n) + np.random.randn(n) * 0.05
    thr    = 0.5
    fc     = 140

    lt = detection_lead_time(scores, thr, fc)
    print(f"Detection lead time: {lt} cycles")

    y_pred = binary_labels(scores, thr)
    y_true = np.zeros(n, dtype=int)
    y_true[fc - 30:] = 1
    prf = precision_recall_f1(y_pred, y_true)
    print(f"Precision: {prf['precision']:.3f}  "
          f"Recall: {prf['recall']:.3f}  "
          f"F1: {prf['f1']:.3f}")
