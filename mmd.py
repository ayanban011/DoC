"""
Maximum Mean Discrepancy (MMD) baseline.

From the paper (Eq. 9 / Supplementary Sec. 10):

    MMD(B', T') = || (1/|B'|) Σ_{x ∈ B'} F'(x)
                   − (1/|T'|) Σ_{x ∈ T'} F'(x) ||₂

This is the *linear* (first-order) MMD estimator, equivalent to the L2
distance between empirical means in feature space.  More powerful kernel-based
MMD estimators exist but require O(N²) computation and are not used in the
paper.

MMD is commonly minimised in domain-adaptation methods (e.g., Deep Domain
Confusion, DAN) and has been shown to correlate with target domain accuracy in
some settings (Long et al., 2017).  However, the paper demonstrates that it
does not reliably predict natural distribution shifts when calibrated on
synthetic ones.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from method import BaseMethod


def _linear_mmd(feats_src: np.ndarray, feats_tgt: np.ndarray) -> float:
    """
    L2-norm of the difference between empirical means in feature space.

        MMD(B, T) = || μ_B − μ_T ||₂
    """
    mu_src = feats_src.mean(axis=0)
    mu_tgt = feats_tgt.mean(axis=0)
    return float(np.linalg.norm(mu_src - mu_tgt))


def _kernel_mmd(
    feats_src: np.ndarray,
    feats_tgt: np.ndarray,
    bandwidth: float | None = None,
) -> float:
    """
    Unbiased kernel MMD with RBF kernel (O(N²), use only for small N).

    Parameters
    ----------
    bandwidth : RBF kernel bandwidth σ.  If None, uses the median heuristic.
    """
    n, m = len(feats_src), len(feats_tgt)

    def rbf(X: np.ndarray, Y: np.ndarray, bw: float) -> np.ndarray:
        diff  = X[:, None, :] - Y[None, :, :]         # (N, M, D)
        dists = (diff ** 2).sum(-1)                    # (N, M)
        return np.exp(-dists / (2 * bw ** 2))

    if bandwidth is None:
        # Median heuristic
        all_feats = np.vstack([feats_src, feats_tgt])
        pairwise  = np.sum((all_feats[:, None] - all_feats[None, :]) ** 2, axis=-1)
        bandwidth = float(np.sqrt(np.median(pairwise) / 2.0))

    K_ss = rbf(feats_src, feats_src, bandwidth)
    K_tt = rbf(feats_tgt, feats_tgt, bandwidth)
    K_st = rbf(feats_src, feats_tgt, bandwidth)

    mmd  = K_ss.sum() / (n * (n - 1)) + K_tt.sum() / (m * (m - 1)) \
           - 2 * K_st.mean()
    return float(np.sqrt(max(mmd, 0.0)))


class MaximumMeanDiscrepancy(BaseMethod):
    """
    Maximum Mean Discrepancy (MMD).

    By default uses the fast linear estimator (L2 distance of feature means),
    matching the paper.  Set ``kernel=True`` for the unbiased RBF-kernel
    estimator (slower, use only for small datasets).
    """

    name = "MMD"
    is_direct_estimator = False
    predicts_gap        = True

    def __init__(self, kernel: bool = False, bandwidth: float | None = None) -> None:
        self.kernel    = kernel
        self.bandwidth = bandwidth

    def compute(
        self,
        source: Dict[str, np.ndarray],
        target: Dict[str, np.ndarray],
    ) -> float:
        """Return MMD between source and target penultimate features."""
        if self.kernel:
            return _kernel_mmd(source["feats"], target["feats"], self.bandwidth)
        return _linear_mmd(source["feats"], target["feats"])
