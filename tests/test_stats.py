"""Tests for the statistics utilities: bootstrap CIs, McNemar, calibration."""
import numpy as np
from sklearn.metrics import accuracy_score

from papavision.stats import bootstrap_metric_ci, expected_calibration_error, mcnemar_test


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, 200)
    y_pred = y_true.copy()
    flip = rng.choice(200, 30, replace=False)
    y_pred[flip] = (y_pred[flip] + 1) % 3  # ~85% accuracy

    res = bootstrap_metric_ci(y_true, y_pred, accuracy_score, n_boot=500, seed=0)
    assert res["ci_low"] <= res["point"] <= res["ci_high"]
    assert 0.0 <= res["ci_low"] < res["ci_high"] <= 1.0


def test_mcnemar_detects_clear_difference():
    # Model A correct on all; Model B wrong on a third — strongly discordant.
    n = 90
    y_true = np.zeros(n, dtype=int)
    pred_a = np.zeros(n, dtype=int)            # all correct
    pred_b = np.zeros(n, dtype=int)
    pred_b[:30] = 1                            # 30 wrong
    res = mcnemar_test(y_true, pred_a, pred_b)
    assert res["b"] == 30 and res["c"] == 0
    assert res["p_value"] < 0.01


def test_mcnemar_identical_models_not_significant():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    pred = np.array([0, 1, 1, 0, 1, 2])
    res = mcnemar_test(y_true, pred, pred)     # identical predictions
    assert res["p_value"] == 1.0


def test_ece_perfectly_calibrated_is_low():
    # Confident + correct everywhere -> small calibration error.
    n = 100
    y_true = np.zeros(n, dtype=int)
    probs = np.zeros((n, 3))
    probs[:, 0] = 0.99
    probs[:, 1] = 0.005
    probs[:, 2] = 0.005
    cal = expected_calibration_error(probs, y_true, n_bins=10)
    assert cal["ece"] < 0.05


def test_ece_overconfident_wrong_is_high():
    # Always 99% confident but always wrong -> ECE near 1.
    n = 100
    y_true = np.ones(n, dtype=int)
    probs = np.zeros((n, 3))
    probs[:, 0] = 0.99
    probs[:, 1] = 0.005
    probs[:, 2] = 0.005
    cal = expected_calibration_error(probs, y_true, n_bins=10)
    assert cal["ece"] > 0.9
