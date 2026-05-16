"""
tests/test_metrics.py
---------------------
Pytest suite for faultsense metrics — all pure functions, no model needed.

Run:
  pytest tests/test_metrics.py -v
  pytest tests/test_metrics.py -v --cov=metrics
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics import (
    rmse,
    nasa_score,
    mean_absolute_error,
    binary_labels,
    binary_labels_persistent,
    detection_lead_time,
    precision_recall_f1,
    false_alarm_rate,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def perfect_rul():
    """Predictions that exactly match ground truth."""
    y = np.array([100.0, 80.0, 50.0, 20.0, 5.0])
    return y, y.copy()


@pytest.fixture
def late_predictions():
    """Predictions systematically higher than truth (optimistic / dangerous)."""
    y_true = np.array([50.0, 30.0, 10.0])
    y_pred = np.array([70.0, 50.0, 30.0])   # all 20 cycles too late
    return y_pred, y_true


@pytest.fixture
def early_predictions():
    """Predictions systematically lower than truth (pessimistic / safe)."""
    y_true = np.array([50.0, 30.0, 10.0])
    y_pred = np.array([30.0, 10.0, 0.0])    # all 20 cycles too early
    return y_pred, y_true


@pytest.fixture
def degradation_scores():
    """Realistic score stream: flat healthy zone then rising degradation."""
    np.random.seed(42)
    healthy    = np.random.uniform(0.05, 0.15, 100)
    degraded   = np.linspace(0.20, 0.90, 100)
    return np.concatenate([healthy, degraded])


# ── RMSE ──────────────────────────────────────────────────────────────────────

class TestRMSE:

    def test_perfect_predictions_give_zero(self, perfect_rul):
        y_pred, y_true = perfect_rul
        assert rmse(y_pred, y_true) == pytest.approx(0.0)

    def test_known_value(self):
        # errors = [3, 4] → MSE = (9+16)/2 = 12.5 → RMSE = √12.5
        y_pred = np.array([3.0, 4.0])
        y_true = np.array([0.0, 0.0])
        assert rmse(y_pred, y_true) == pytest.approx(np.sqrt(12.5))

    def test_symmetric(self):
        y_pred = np.array([10.0, 20.0])
        y_true = np.array([15.0, 25.0])
        # RMSE(pred, true) == RMSE(true, pred) because errors are squared
        assert rmse(y_pred, y_true) == pytest.approx(rmse(y_true, y_pred))

    def test_returns_float(self):
        assert isinstance(rmse(np.array([1.0]), np.array([2.0])), float)

    def test_single_sample(self):
        assert rmse(np.array([10.0]), np.array([7.0])) == pytest.approx(3.0)


# ── NASA Score ────────────────────────────────────────────────────────────────

class TestNASAScore:

    def test_perfect_predictions_give_zero(self, perfect_rul):
        y_pred, y_true = perfect_rul
        assert nasa_score(y_pred, y_true) == pytest.approx(0.0, abs=1e-6)

    def test_late_penalised_harder_than_early(self, late_predictions,
                                               early_predictions):
        """Late predictions (optimistic) must be penalised more than early."""
        late_score  = nasa_score(*late_predictions)
        early_score = nasa_score(*early_predictions)
        assert late_score > early_score, (
            f"Late score {late_score:.4f} should exceed early score {early_score:.4f}"
        )

    def test_score_increases_with_error_magnitude(self):
        y_true = np.array([50.0])
        small_error = nasa_score(np.array([60.0]), y_true)   # d = +10
        large_error = nasa_score(np.array([80.0]), y_true)   # d = +30
        assert large_error > small_error

    def test_symmetric_constants_give_symmetric_score(self):
        """When c_early == c_late the function is symmetric around zero."""
        y_true = np.array([50.0])
        late  = nasa_score(np.array([60.0]), y_true, c_early=10, c_late=10)
        early = nasa_score(np.array([40.0]), y_true, c_early=10, c_late=10)
        assert late == pytest.approx(early, rel=1e-5)

    def test_returns_float(self):
        assert isinstance(nasa_score(np.array([1.0]), np.array([1.0])), float)

    def test_sum_over_engines(self):
        """NASA score is a SUM not a mean — more engines = larger value."""
        y_true_1 = np.array([50.0])
        y_true_2 = np.array([50.0, 50.0])
        y_pred_1 = np.array([60.0])
        y_pred_2 = np.array([60.0, 60.0])
        score_1 = nasa_score(y_pred_1, y_true_1)
        score_2 = nasa_score(y_pred_2, y_true_2)
        assert score_2 == pytest.approx(2 * score_1)


# ── MAE ───────────────────────────────────────────────────────────────────────

class TestMAE:

    def test_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mean_absolute_error(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_pred = np.array([1.0, 3.0])
        y_true = np.array([0.0, 0.0])
        assert mean_absolute_error(y_pred, y_true) == pytest.approx(2.0)

    def test_mae_le_rmse_for_unequal_errors(self):
        """MAE ≤ RMSE always (RMSE amplifies large errors)."""
        y_pred = np.array([1.0, 10.0, 1.0])
        y_true = np.array([0.0,  0.0, 0.0])
        assert mean_absolute_error(y_pred, y_true) <= rmse(y_pred, y_true)


# ── Binary Labels ─────────────────────────────────────────────────────────────

class TestBinaryLabels:

    def test_all_below_threshold(self):
        scores = np.array([0.1, 0.2, 0.3])
        result = binary_labels(scores, threshold=0.5)
        assert result.sum() == 0

    def test_all_above_threshold(self):
        scores = np.array([0.6, 0.7, 0.8])
        result = binary_labels(scores, threshold=0.5)
        assert result.sum() == 3

    def test_boundary_is_exclusive(self):
        """Score exactly equal to threshold should NOT trigger alarm."""
        scores = np.array([0.5])
        result = binary_labels(scores, threshold=0.5)
        assert result[0] == 0

    def test_output_dtype_is_int(self):
        scores = np.array([0.1, 0.9])
        result = binary_labels(scores, threshold=0.5)
        assert result.dtype in (np.int32, np.int64, int)

    def test_mixed(self):
        scores = np.array([0.3, 0.7, 0.4, 0.8])
        result = binary_labels(scores, threshold=0.5)
        np.testing.assert_array_equal(result, [0, 1, 0, 1])


# ── Binary Labels Persistent ──────────────────────────────────────────────────

class TestBinaryLabelsPersistent:

    def test_single_spike_suppressed(self):
        """One cycle above threshold should not trigger alarm (min_consecutive=3)."""
        scores = np.array([0.1, 0.9, 0.1, 0.1, 0.1])
        result = binary_labels_persistent(scores, threshold=0.5, min_consecutive=3)
        assert result.sum() == 0

    def test_sustained_alarm_fires(self):
        """Three consecutive cycles above threshold should trigger alarm."""
        scores = np.array([0.1, 0.9, 0.9, 0.9, 0.9])
        result = binary_labels_persistent(scores, threshold=0.5, min_consecutive=3)
        # alarm should fire from index 3 onwards (after 3 consecutive)
        assert result[3] == 1
        assert result[4] == 1

    def test_alarm_resets_after_drop(self):
        """Score dropping below threshold resets the counter."""
        scores = np.array([0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9])
        result = binary_labels_persistent(scores, threshold=0.5, min_consecutive=3)
        # first two cycles: counter=2, not enough
        assert result[0] == 0
        assert result[1] == 0
        # after reset at index 2, alarm fires at index 5 (3 consecutive: 3,4,5)
        assert result[5] == 1

    def test_min_consecutive_one_equals_binary_labels(self):
        """min_consecutive=1 should behave identically to binary_labels."""
        scores = np.array([0.3, 0.7, 0.4, 0.8, 0.6])
        simple     = binary_labels(scores, threshold=0.5)
        persistent = binary_labels_persistent(scores, threshold=0.5,
                                               min_consecutive=1)
        np.testing.assert_array_equal(simple, persistent)


# ── Detection Lead Time ───────────────────────────────────────────────────────

class TestDetectionLeadTime:

    def test_early_detection(self):
        """Detector fires before fault — lead time should be positive."""
        # Deterministic: healthy zone then crosses 0.5 at cycle 80, fault at 100
        scores = np.concatenate([
            np.full(80, 0.1),          # healthy
            np.linspace(0.5, 0.9, 20), # rising — first value is exactly 0.5
            np.full(100, 0.9),         # degraded
        ])
        # threshold=0.49 so 0.5 triggers, fault at cycle 100
        lt = detection_lead_time(scores, threshold=0.49, true_fault_cycle=100)
        assert lt is not None
        assert lt > 0

    def test_missed_detection_returns_none(self):
        """If score never crosses threshold, return None."""
        scores = np.full(100, 0.1)   # always below threshold
        lt = detection_lead_time(scores, threshold=0.5, true_fault_cycle=80)
        assert lt is None

    def test_lead_time_calculation(self):
        """Alarm at cycle 70, fault at cycle 100 → lead time = 30."""
        scores = np.concatenate([np.zeros(70), np.ones(30)])
        lt = detection_lead_time(scores, threshold=0.5, true_fault_cycle=100)
        assert lt == 30

    def test_negative_lead_time_late_detection(self):
        """Alarm after fault onset → negative lead time."""
        scores = np.concatenate([np.zeros(110), np.ones(20)])
        lt = detection_lead_time(scores, threshold=0.5, true_fault_cycle=100)
        assert lt is not None
        assert lt < 0


# ── Precision / Recall / F1 ───────────────────────────────────────────────────

class TestPrecisionRecallF1:

    def test_perfect_classifier(self):
        y_pred = np.array([0, 0, 1, 1])
        y_true = np.array([0, 0, 1, 1])
        result = precision_recall_f1(y_pred, y_true)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"]    == pytest.approx(1.0)
        assert result["f1"]        == pytest.approx(1.0)

    def test_all_false_positives(self):
        y_pred = np.array([1, 1, 1, 1])
        y_true = np.array([0, 0, 0, 0])
        result = precision_recall_f1(y_pred, y_true)
        assert result["precision"] == pytest.approx(0.0)

    def test_all_false_negatives(self):
        y_pred = np.array([0, 0, 0, 0])
        y_true = np.array([1, 1, 1, 1])
        result = precision_recall_f1(y_pred, y_true)
        assert result["recall"] == pytest.approx(0.0)

    def test_f1_is_harmonic_mean(self):
        y_pred = np.array([1, 1, 0, 1])
        y_true = np.array([1, 0, 1, 1])
        result = precision_recall_f1(y_pred, y_true)
        p, r = result["precision"], result["recall"]
        expected_f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        assert result["f1"] == pytest.approx(expected_f1, rel=1e-5)

    def test_returns_dict_with_required_keys(self):
        result = precision_recall_f1(np.array([1]), np.array([1]))
        assert all(k in result for k in ("precision", "recall", "f1"))


# ── False Alarm Rate ──────────────────────────────────────────────────────────

class TestFalseAlarmRate:

    def test_no_false_alarms(self):
        """Detector silent in healthy zone → FAR = 0."""
        scores = np.concatenate([np.zeros(100), np.ones(50)])
        far = false_alarm_rate([scores], threshold=0.5,
                                fault_cycles=[100], fault_window=30)
        assert far == pytest.approx(0.0)

    def test_all_healthy_cycles_alarmed(self):
        """Detector always on → FAR = 100 (alarms per 100 healthy cycles)."""
        scores = np.ones(100)   # always above threshold
        # fault at cycle 90, window=30 → healthy zone = cycles 0-59
        far = false_alarm_rate([scores], threshold=0.5,
                                fault_cycles=[90], fault_window=30)
        assert far == pytest.approx(100.0)

    def test_far_is_non_negative(self, degradation_scores):
        far = false_alarm_rate([degradation_scores], threshold=0.5,
                                fault_cycles=[120], fault_window=30)
        assert far >= 0.0
