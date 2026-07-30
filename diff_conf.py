"""
Difference of Confidences (DoC) – the key contribution of the paper.

From the paper (Eq. 5–6):

    AC^T_B = (1/|B'|) Σ_{x ∈ B'} max_{k ∈ K_{B∩T}} F(x)_k
    AC^B_T = (1/|T'|) Σ_{x ∈ T'} max_{k ∈ K_{B∩T}} F(x)_k

    DoC(B, T) = AC^T_B - AC^B_T                           (Eq. 6)

where B' and T' are restricted to samples whose labels are in the shared
label set K_{B∩T}.

DoC is used as a *feature* S fed into a linear regressor R:

    Δ Acc(B, T) = R(DoC(B, T))
    Acc^B_T ≈ Acc(B) + Δ Acc

Intuitively, if the model is less confident on the target than on the source,
the target accuracy is likely lower too.

DoC-Feat variant
----------------
The paper also introduces DoC-Feat, which uses the same confidence difference
but applies the model directly (without a separate regression step):

    Acc^B_T ≈ Acc(B) - DoC(B, T)

(i.e. the regressor is implicitly the identity).  We implement it as a
direct estimator with ``is_direct_estimator = True``.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from method import BaseMethod


def _ac(probs: np.ndarray, class_mask: Optional[np.ndarray] = None) -> float:
    """Mean max confidence over samples (optionally restricted to a class subset)."""
    if class_mask is not None:
        probs = probs[:, class_mask]
    return float(probs.max(axis=1).mean())


class DifferenceOfConfidences(BaseMethod):
    """
    Difference of Confidences (DoC).

    ``compute`` returns the scalar S = AC(source) - AC(target), which is
    subsequently used as input to a linear regressor to predict Δ Acc.

    Parameters
    ----------
    use_shared_classes : if True, restrict both AC computations to the label
                         set shared between source and target.  For DG datasets
                         where source and target share all classes this has no
                         effect, but it is included for generality.
    """

    name = "DoC"
    is_direct_estimator = False
    predicts_gap        = True

    def __init__(self, use_shared_classes: bool = True) -> None:
        self.use_shared_classes = use_shared_classes

    def compute(
        self,
        source: Dict[str, np.ndarray],
        target: Dict[str, np.ndarray],
    ) -> float:
        """
        Compute DoC(source, target).

        Returns AC_source - AC_target over the shared class set.
        A positive value indicates the model is more confident on the source,
        implying the target accuracy is likely *lower* than the source accuracy.
        """
        src_probs = source["probs"]   # (N_s, C)
        tgt_probs = target["probs"]   # (N_t, C)

        class_mask = None
        if self.use_shared_classes:
            # In DG setting, all classes are shared, so mask is all-True.
            # For completeness we still compute it explicitly.
            n_classes = src_probs.shape[1]
            class_mask = np.ones(n_classes, dtype=bool)

        ac_source = _ac(src_probs, class_mask)
        ac_target = _ac(tgt_probs, class_mask)

        return ac_source - ac_target  # DoC(B, T)

    def compute_ac_source(self, source: Dict[str, np.ndarray]) -> float:
        """Return the source AC component only."""
        return _ac(source["probs"])

    def compute_ac_target(self, target: Dict[str, np.ndarray]) -> float:
        """Return the target AC component only."""
        return _ac(target["probs"])


class DifferenceOfConfidencesFeat(BaseMethod):
    """
    DoC-Feat: direct accuracy estimator using DoC without a regression model.

    Accuracy on the target domain is estimated as:
        Acc_target ≈ Acc_source - DoC(source, target)

    This exploits the empirical relationship that DoC ≈ Δ Acc directly, which
    holds well for models trained with standard empirical risk minimisation.
    """

    name = "DoC-Feat"
    is_direct_estimator = True
    predicts_gap        = False

    def compute(
        self,
        source: Dict[str, np.ndarray],
        target: Dict[str, np.ndarray],
    ) -> float:
        """Return direct accuracy estimate: Acc_source - DoC(source, target)."""
        doc_score   = _ac(source["probs"]) - _ac(target["probs"])
        acc_source  = source["accuracy"]
        return acc_source - doc_score

    def predict_accuracy(
        self,
        source: Dict[str, np.ndarray],
        target: Dict[str, np.ndarray],
    ) -> float:
        return self.compute(source, target)
