"""
ltc.evaluation.metrics — ground-truth comparison metrics.

All functions operate on numpy arrays and return Python scalars (floats).
They are stateless and can be called independently or via scorer.py.

Available metrics
-----------------
recovery_accuracy(estimated, true)    → float   MAPE of LTC recovery
mape(estimated, true)                 → float   Mean Absolute Percentage Error
mae(estimated, true)                  → float   Mean Absolute Error
rmse(estimated, true)                 → float   Root Mean Squared Error
correlation(estimated, true)          → float   Pearson correlation
ci_coverage(lower, upper, true)       → float   Fraction of weeks where true ∈ [lower, upper]
bias(estimated, true)                 → float   Signed mean error (positive = over-estimate)
total_recovery_ratio(estimated, true) → float   Sum(estimated) / Sum(true)
"""

import numpy as np


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise division; returns 0.0 where denominator is zero."""
    denom = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
    return numerator / denom


def mape(estimated: np.ndarray, true: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.

    MAPE = mean( |estimated - true| / max(|true|, ε) ) × 100

    Returns a percentage (e.g., 15.3 means 15.3% error).
    A return value of 0.0 indicates perfect recovery.

    Parameters
    ----------
    estimated : np.ndarray (T,)
        Model-estimated values.
    true : np.ndarray (T,)
        Ground-truth values.

    Returns
    -------
    float — MAPE in percent.
    """
    estimated = np.asarray(estimated, dtype=float)
    true = np.asarray(true, dtype=float)
    ape = np.abs(_safe_divide(estimated - true, true))
    return float(np.mean(ape) * 100.0)


def recovery_accuracy(estimated: np.ndarray, true: np.ndarray) -> float:
    """
    Recovery accuracy: complement of MAPE, capped at 100%.

    recovery_accuracy = max(0, 100 - MAPE)

    Interpretation:
      100 = perfect recovery
        0 = estimates are 100% off (or worse)
    """
    return float(max(0.0, 100.0 - mape(estimated, true)))


def mae(estimated: np.ndarray, true: np.ndarray) -> float:
    """Mean Absolute Error in the same units as the input ($M)."""
    return float(np.mean(np.abs(np.asarray(estimated) - np.asarray(true))))


def rmse(estimated: np.ndarray, true: np.ndarray) -> float:
    """Root Mean Squared Error in the same units as the input ($M)."""
    diff = np.asarray(estimated, dtype=float) - np.asarray(true, dtype=float)
    return float(np.sqrt(np.mean(diff ** 2)))


def correlation(estimated: np.ndarray, true: np.ndarray) -> float:
    """
    Pearson correlation between estimated and true series.

    Captures shape similarity independent of scale.
    Returns NaN if either series has zero variance.
    """
    estimated = np.asarray(estimated, dtype=float)
    true = np.asarray(true, dtype=float)
    if estimated.std() == 0 or true.std() == 0:
        return float("nan")
    return float(np.corrcoef(estimated, true)[0, 1])


def ci_coverage(
    lower: np.ndarray, upper: np.ndarray, true: np.ndarray
) -> float:
    """
    Credible / confidence interval coverage.

    Fraction of time periods where the true value falls within [lower, upper].
    A well-calibrated 95% CI should achieve ≈ 0.95 coverage.

    Parameters
    ----------
    lower : np.ndarray (T,)   — lower bound of the estimated interval.
    upper : np.ndarray (T,)   — upper bound.
    true  : np.ndarray (T,)   — ground-truth values.

    Returns
    -------
    float — coverage fraction in [0, 1].
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    true = np.asarray(true, dtype=float)
    covered = (true >= lower) & (true <= upper)
    return float(covered.mean())


def bias(estimated: np.ndarray, true: np.ndarray) -> float:
    """
    Signed mean error (estimated − true).

    Positive = model over-estimates LTC.
    Negative = model under-estimates LTC.
    """
    return float(np.mean(np.asarray(estimated, dtype=float) - np.asarray(true, dtype=float)))


def total_recovery_ratio(estimated: np.ndarray, true: np.ndarray) -> float:
    """
    Ratio of total estimated LTC to total true LTC over the full series.

    total_recovery_ratio = sum(estimated) / sum(true)

    1.0 = exact aggregate recovery
    > 1.0 = over-attribution
    < 1.0 = under-attribution

    The key summary statistic for the paper's benchmark tables.
    """
    estimated = np.asarray(estimated, dtype=float)
    true = np.asarray(true, dtype=float)
    total_true = true.sum()
    if abs(total_true) < 1e-10:
        return float("nan")
    return float(estimated.sum() / total_true)


def compute_all_metrics(
    estimated: np.ndarray,
    true: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> dict:
    """
    Compute the full set of metrics for one (estimated, true) pair.

    Parameters
    ----------
    estimated : np.ndarray (T,)
    true : np.ndarray (T,)
    lower : np.ndarray (T,) or None — lower CI bound (optional)
    upper : np.ndarray (T,) or None — upper CI bound (optional)

    Returns
    -------
    dict with keys: mape, recovery_accuracy, mae, rmse, correlation,
                    bias, total_recovery_ratio, [ci_coverage if CI provided]
    """
    result = {
        "mape": mape(estimated, true),
        "recovery_accuracy": recovery_accuracy(estimated, true),
        "mae": mae(estimated, true),
        "rmse": rmse(estimated, true),
        "correlation": correlation(estimated, true),
        "bias": bias(estimated, true),
        "total_recovery_ratio": total_recovery_ratio(estimated, true),
    }
    if lower is not None and upper is not None:
        result["ci_coverage"] = ci_coverage(lower, upper, true)
    return result
