"""
Evaluation metrics for accuracy prediction.

Primary metric (paper): Mean Absolute Error (MAE) between predicted and
actual accuracy on the target domain.  Lower is better.

Secondary metrics: Pearson correlation ρ, MSE, and bootstrap confidence
intervals on MAE (95 %).

The paper reports MAE ± std where std is computed across model architectures.
In the DG setting we report MAE ± std across target domains and multiple
random seeds.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean Absolute Error."""
    a, b = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    return float(np.abs(a - b).mean())


def mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean Squared Error."""
    a, b = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    return float(((a - b) ** 2).mean())


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def pearson(y_true: Sequence[float], y_pred: Sequence[float]) -> Tuple[float, float]:
    """Pearson r and p-value."""
    if len(set(y_true)) < 2 or len(set(y_pred)) < 2:
        return 0.0, 1.0
    r, p = pearsonr(y_true, y_pred)
    return float(r), float(p)


def spearman(y_true: Sequence[float], y_pred: Sequence[float]) -> Tuple[float, float]:
    """Spearman ρ and p-value."""
    if len(set(y_true)) < 2 or len(set(y_pred)) < 2:
        return 0.0, 1.0
    r, p = spearmanr(y_true, y_pred)
    return float(r), float(p)


def r_squared(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Coefficient of determination R²."""
    a, b = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    ss_res = ((a - b) ** 2).sum()
    ss_tot = ((a - a.mean()) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def bootstrap_mae_ci(
    y_true:      Sequence[float],
    y_pred:      Sequence[float],
    n_bootstrap: int   = 1000,
    alpha:       float = 0.05,
    seed:        int   = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap 95 % confidence interval around MAE.

    Returns
    -------
    (mae_mean, ci_lower, ci_upper)
    """
    rng  = np.random.default_rng(seed)
    a    = np.array(y_true, dtype=float)
    b    = np.array(y_pred, dtype=float)
    n    = len(a)

    boot_maes = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_maes.append(float(np.abs(a[idx] - b[idx]).mean()))

    lo = float(np.percentile(boot_maes, 100 * alpha / 2))
    hi = float(np.percentile(boot_maes, 100 * (1 - alpha / 2)))
    return float(np.mean(boot_maes)), lo, hi


def summarise_results(
    method_name:    str,
    actual_accs:    Sequence[float],
    predicted_accs: Sequence[float],
) -> Dict[str, float]:
    """
    Return a dict of all evaluation metrics for one method.

    Parameters
    ----------
    method_name    : label for the method
    actual_accs    : ground-truth target accuracies
    predicted_accs : model-predicted target accuracies

    Returns
    -------
    dict with keys: method, mae, mse, rmse, r2, pearson_r, pearson_p,
                    spearman_r, spearman_p, ci_lower, ci_upper
    """
    mae_val          = mae(actual_accs, predicted_accs)
    mse_val          = mse(actual_accs, predicted_accs)
    rmse_val         = rmse(actual_accs, predicted_accs)
    r2_val           = r_squared(actual_accs, predicted_accs)
    pear_r, pear_p   = pearson(actual_accs, predicted_accs)
    spear_r, spear_p = spearman(actual_accs, predicted_accs)
    _, ci_lo, ci_hi  = bootstrap_mae_ci(actual_accs, predicted_accs)

    return {
        "method":      method_name,
        "mae":         mae_val,
        "mse":         mse_val,
        "rmse":        rmse_val,
        "r2":          r2_val,
        "pearson_r":   pear_r,
        "pearson_p":   pear_p,
        "spearman_r":  spear_r,
        "spearman_p":  spear_p,
        "ci_lower":    ci_lo,
        "ci_upper":    ci_hi,
    }


def relative_improvement(baseline_mae: float, method_mae: float) -> float:
    """
    Percentage reduction in MAE versus baseline.
    Positive = better than baseline; negative = worse.
    """
    if baseline_mae == 0:
        return 0.0
    return float((baseline_mae - method_mae) / baseline_mae * 100)
