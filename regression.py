"""
Linear (and non-linear) regression for accuracy prediction.

The paper (Sec. 3) describes a regression model R that maps the distributional
similarity score S to the accuracy gap Δ Acc:

    min_{R}  Σ_{T ∈ C}  ||R(S_{B,T}) − Δ Acc(B, T)||²

where C is the calibration set.  At test time:

    Acc_T ≈ Acc_B + R(S_{B,T})

Primary model: ``sklearn.linear_model.LinearRegression``
Secondary:     3-layer MLP (Supp. Sec. 7): 512 → 256 → 128 → 1

Both are wrapped in ``AccuracyPredictor`` which handles fitting, prediction,
and serialisation.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AccuracyPredictor:
    """
    Wrapper around a regression model that:
    1. Fits on calibration (S, Δ Acc) pairs.
    2. Predicts Δ Acc on unseen target domains.
    3. Converts predicted Δ Acc back to absolute accuracy.

    Parameters
    ----------
    model_type  : ``"linear"`` | ``"ridge"`` | ``"mlp"``
    alpha       : regularisation strength for Ridge regression (ignored for
                  linear and MLP)
    scale_input : if True, standardise the scalar input S before regression
                  (useful for MLP and Ridge)
    """

    def __init__(
        self,
        model_type:  str   = "linear",
        alpha:       float = 1.0,
        scale_input: bool  = False,
    ) -> None:
        self.model_type  = model_type
        self.alpha       = alpha
        self.scale_input = scale_input
        self._fitted     = False

        if model_type == "linear":
            self._reg = LinearRegression()
        elif model_type == "ridge":
            self._reg = Ridge(alpha=alpha)
        elif model_type == "mlp":
            # Architecture from the paper's supplementary material
            self._reg = MLPRegressor(
                hidden_layer_sizes=(512, 256, 128),
                activation="relu",
                max_iter=20_000,
                tol=1e-5,
                learning_rate_init=1e-4,
                alpha=1e-3,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type!r}")

        self._scaler = StandardScaler() if scale_input else None

    # ── Fitting ──────────────────────────────────────────────────────────────

    def fit(
        self,
        scores:    Sequence[float],
        delta_acc: Sequence[float],
    ) -> "AccuracyPredictor":
        """
        Fit the regressor on calibration (S, Δ Acc) pairs.

        Parameters
        ----------
        scores    : list of scalar similarity/distance scores S
        delta_acc : list of corresponding Δ Acc = Acc_source − Acc_target
                    (positive when source > target)
        """
        X = np.array(scores,    dtype=np.float64).reshape(-1, 1)
        y = np.array(delta_acc, dtype=np.float64)

        if self._scaler is not None:
            X = self._scaler.fit_transform(X)

        self._reg.fit(X, y)
        self._fitted = True

        # Log calibration R² for sanity checking
        y_pred = self._reg.predict(X)
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        logger.info(f"[AccuracyPredictor] Calibration R² = {r2:.4f}  (n={len(y)})")
        return self

    # ── Prediction ──────────────────────────────────────────────────────────

    def predict_gap(self, score: float) -> float:
        """Predict Δ Acc = Acc_source − Acc_target from a single score S."""
        self._check_fitted()
        X = np.array([[score]], dtype=np.float64)
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return float(self._reg.predict(X)[0])

    def predict_accuracy(self, score: float, source_accuracy: float) -> float:
        """
        Predict absolute target accuracy.

        Acc_target ≈ Acc_source − R(S)
        (The regressor predicts Acc_source − Acc_target, so we subtract.)
        """
        delta = self.predict_gap(score)
        return source_accuracy - delta

    # ── Serialisation ────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "AccuracyPredictor":
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── Utilities ────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "AccuracyPredictor has not been fitted yet. Call .fit() first."
            )

    @property
    def coef_(self) -> Optional[np.ndarray]:
        if hasattr(self._reg, "coef_"):
            return self._reg.coef_
        return None

    @property
    def intercept_(self):
        if hasattr(self._reg, "intercept_"):
            return self._reg.intercept_
        return None


# ── Cross-validated calibration ──────────────────────────────────────────────

def cross_validate_predictor(
    scores:     Sequence[float],
    delta_acc:  Sequence[float],
    n_splits:   int = 5,
    model_type: str = "linear",
) -> Tuple[float, float]:
    """
    K-fold cross-validation of the regression model on calibration data.

    Returns
    -------
    mean_mae : float – mean MAE across folds
    std_mae  : float – std of MAE across folds
    """
    from sklearn.model_selection import KFold

    scores_arr    = np.array(scores,    dtype=np.float64)
    delta_acc_arr = np.array(delta_acc, dtype=np.float64)

    kf   = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    maes = []

    for tr_idx, va_idx in kf.split(scores_arr):
        pred = AccuracyPredictor(model_type=model_type)
        pred.fit(scores_arr[tr_idx], delta_acc_arr[tr_idx])
        preds    = [pred.predict_gap(s) for s in scores_arr[va_idx]]
        mae      = float(np.abs(np.array(preds) - delta_acc_arr[va_idx]).mean())
        maes.append(mae)

    return float(np.mean(maes)), float(np.std(maes))
