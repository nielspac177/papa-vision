"""Statistical tools for honest model comparison.

* ``bootstrap_metric_ci`` — non-parametric 95% confidence intervals for any
  classification metric, via case resampling of the test set.
* ``mcnemar_test`` — the appropriate *paired* test for comparing two classifiers
  on the **same** test set (exact binomial for small discordant counts, otherwise
  a continuity-corrected chi-square).
* ``expected_calibration_error`` — ECE plus per-bin data for reliability diagrams.

Keeping these in one place (rather than scattering ad-hoc computations through the
evaluation code) makes the statistical methodology auditable in a single file.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Bootstrap confidence intervals
# --------------------------------------------------------------------------- #
def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI for a metric computed from ``(y_true, y_pred)``.

    We resample test *cases* (with replacement) ``n_boot`` times, recompute the
    metric on each resample, and report the point estimate plus the empirical
    ``[alpha/2, 1-alpha/2]`` percentiles. This captures the test-set sampling
    uncertainty that a single accuracy number hides.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    point = float(metric_fn(y_true, y_pred))
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = metric_fn(y_true[idx], y_pred[idx])

    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {"point": point, "ci_low": lo, "ci_high": hi, "n_boot": n_boot}


# --------------------------------------------------------------------------- #
# McNemar's paired test
# --------------------------------------------------------------------------- #
def mcnemar_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> dict[str, float]:
    """McNemar's test comparing two classifiers on the same labelled test set.

    Builds the discordant counts:

    * ``b`` = #(A correct, B wrong)
    * ``c`` = #(A wrong,  B correct)

    For small ``b + c`` (< 25) the exact two-sided binomial test is used; otherwise
    a chi-square statistic with Edwards' continuity correction. A small p-value
    means the two models differ significantly in error pattern, not by chance.
    """
    y_true = np.asarray(y_true)
    a_correct = np.asarray(pred_a) == y_true
    b_correct = np.asarray(pred_b) == y_true

    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    n_discordant = b + c

    if n_discordant == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "method": "degenerate"}

    if n_discordant < 25:
        # Exact two-sided binomial test against p=0.5.
        res = stats.binomtest(min(b, c), n_discordant, 0.5, alternative="two-sided")
        return {
            "b": b, "c": c,
            "statistic": float(min(b, c)),
            "p_value": float(res.pvalue),
            "method": "exact_binomial",
        }

    stat = (abs(b - c) - 1.0) ** 2 / n_discordant  # continuity-corrected chi-square
    p = float(stats.chi2.sf(stat, df=1))
    return {"b": b, "c": c, "statistic": float(stat), "p_value": p, "method": "chi2_cc"}


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def expected_calibration_error(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Expected Calibration Error (ECE) with per-bin reliability data.

    Confidence = max predicted probability; accuracy = fraction correct within the
    bin. ECE is the sample-weighted average gap between confidence and accuracy
    across ``n_bins`` equal-width confidence bins. Lower is better; a perfectly
    calibrated model has ECE = 0.
    """
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    bin_conf, bin_acc, bin_count = [], [], []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        # Include the right edge in the final bin.
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences <= hi)
        count = int(np.sum(in_bin))
        if count > 0:
            avg_conf = float(np.mean(confidences[in_bin]))
            avg_acc = float(np.mean(accuracies[in_bin]))
            ece += (count / n) * abs(avg_conf - avg_acc)
        else:
            avg_conf, avg_acc = (lo + hi) / 2, 0.0
        bin_conf.append(avg_conf)
        bin_acc.append(avg_acc)
        bin_count.append(count)

    return {
        "ece": float(ece),
        "bin_edges": bins.tolist(),
        "bin_confidence": bin_conf,
        "bin_accuracy": bin_acc,
        "bin_count": bin_count,
        "n_bins": n_bins,
    }
