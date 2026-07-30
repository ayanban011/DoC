"""
Difference of (average) Entropy (DoE).

From the paper (Eq. 7–8):

    Ent^T_B = -(1/|B'|) Σ_{x ∈ B'} Σ_{k ∈ K_{B∩T}} F(x)_k log F(x)_k
    Ent^B_T = -(1/|T'|) Σ_{x ∈ T'} Σ_{k ∈ K_{B∩T}} F(x)_k log F(x)_k

    DoE(B, T) = Ent^T_B - Ent^B_T                         (Eq. 8)

DoE shares the same prediction framework as DoC (linear regressor on the
scalar feature S), but uses average entropy instead of average max-confidence.

Entropy and max-confidence are inversely related for unimodal distributions,
so DoE and DoC encode similar information.  Empirically DoC outperforms DoE
on most benchmarks (the paper attributes this to DoE retaining "suboptimal"
probability mass from non-maximum classes), but DoE is still substantially
better than distance-based baselines.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from method import BaseMethod


_EPS = 1e-12   # numerical safety for log


def _entropy(probs: np.ndarray, class_mask: Optional[np.ndarray] = None) -> float:
    """
    Mean Shannon entropy (nats) over all samples in a dataset.

    H(x) = -Σ_k p_k log(p_k)
    """
    if class_mask is not None:
        probs = probs[:, class_mask]
        # Re-normalise so probabilities sum to 1 over the restricted classes
        row_sums = probs.sum(axis=1, keepdims=True)
        probs    = probs / np.maximum(row_sums, _EPS)

    ent_per_sample = -(probs * np.log(probs + _EPS)).sum(axis=1)  # (N,)
    return float(ent_per_sample.mean())


class DifferenceOfEntropy(BaseMethod):
    """
    Difference of (average) Entropy (DoE).

    ``compute`` returns S = Ent(source) - Ent(target).

    A positive value means the model is *more uncertain* on source than
    target, implying a potential accuracy *drop* on target is less severe.
    (Note: higher entropy → lower confidence → typically lower accuracy,
     so DoE and DoC have the same sign convention for Δ Acc prediction.)
    """

    name = "DoE"
    is_direct_estimator = False
    predicts_gap        = True

    def __init__(self, use_shared_classes: bool = True) -> None:
        self.use_shared_classes = use_shared_classes

    def compute(
        self,
        source: Dict[str, np.ndarray],
        target: Dict[str, np.ndarray],
    ) -> float:
        """Compute DoE(source, target) = Ent(source) - Ent(target)."""
        src_probs = source["probs"]
        tgt_probs = target["probs"]

        class_mask = None
        if self.use_shared_classes:
            n_classes  = src_probs.shape[1]
            class_mask = np.ones(n_classes, dtype=bool)

        ent_source = _entropy(src_probs, class_mask)
        ent_target = _entropy(tgt_probs, class_mask)

        return ent_source - ent_target

    def entropy_source(self, source: Dict[str, np.ndarray]) -> float:
        return _entropy(source["probs"])

    def entropy_target(self, target: Dict[str, np.ndarray]) -> float:
        return _entropy(target["probs"])
