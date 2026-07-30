"""
Average Confidence (AC) baseline.

From the paper (Eq. 4):

    AC_T = (1 / |T|) Σ_{x ∈ T}  max_{k ∈ K_{B∩T}}  F(x)_k

AC directly estimates the *accuracy* on the target distribution by taking
the mean max-softmax confidence over target samples.  It is biased upward
for miscalibrated networks (modern networks tend to be over-confident) but
provides a surprisingly competitive baseline, especially for natural shifts.

Because AC predicts absolute accuracy (not a gap), it does **not** use a
regression model.  The predicted accuracy for the target domain is simply AC_T.

Variants
--------
* ``AverageConfidence``          – standard AC over target only
* ``AverageConfidenceTempScaled``– AC after post-hoc temperature scaling
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from method import BaseMethod


def _average_confidence(probs: np.ndarray, class_mask: np.ndarray | None = None) -> float:
    """
    Mean max-probability (confidence) over a dataset.

    Parameters
    ----------
    probs      : (N, C) softmax probabilities
    class_mask : boolean array of length C selecting which classes to consider
                 (shared label space K_{B∩T}).  None = use all classes.
    """
    if class_mask is not None:
        probs = probs[:, class_mask]
    return float(probs.max(axis=1).mean())


class AverageConfidence(BaseMethod):
    """
    Average Confidence (AC).

    Directly estimates the target accuracy as the mean max-softmax
    confidence over unlabelled target samples.  No regressor is needed.

    ``compute`` returns AC_T (a direct accuracy estimate, not a gap).
    """

    name = "AC"
    is_direct_estimator = True
    predicts_gap        = False

    def compute(
        self,
        source: Dict[str, np.ndarray],
        target: Dict[str, np.ndarray],
    ) -> float:
        """Return AC on the target domain."""
        return _average_confidence(target["probs"])

    def predict_accuracy(
        self,
        source: Dict[str, np.ndarray],
        target: Dict[str, np.ndarray],
    ) -> float:
        """Convenience alias – same as ``compute``."""
        return self.compute(source, target)


class AverageConfidenceTempScaled(BaseMethod):
    """
    Temperature-Scaled Average Confidence (AC-TS).

    Same as AC but the model producing ``target["probs"]`` has already been
    post-hoc calibrated via temperature scaling on the source validation set.
    The method itself is identical; calibration is assumed to happen upstream
    (in the trainer / experiment runner).
    """

    name = "AC-TempScaled"
    is_direct_estimator = True
    predicts_gap        = False

    def compute(
        self,
        source: Dict[str, np.ndarray],
        target: Dict[str, np.ndarray],
    ) -> float:
        return _average_confidence(target["probs"])
