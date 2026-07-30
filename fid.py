"""
Fréchet Distance (FD) baseline.

From the paper (Eq. 10 / Supplementary Sec. 10):

    FD(B, T) = ||μ_B - μ_T||²
               + Tr(Σ_B + Σ_T − 2 (Σ_B Σ_T)^{1/2})

where μ and Σ are the empirical mean and covariance of the penultimate-layer
features F'(·) computed over the respective datasets.

This is the same formula as Fréchet Inception Distance (FID; Heusel et al.,
2017) but applied to task-specific features rather than Inception-v3 features.

The paper shows that FD is competitive on *synthetic* distribution shifts but
fails to transfer to natural shifts when the regression model is trained on
synthetic calibration data, performing worse than the AC baseline.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import scipy.linalg

from method import BaseMethod


def _frechet_distance(
    feats_src: np.ndarray,
    feats_tgt: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Compute the Fréchet distance between two feature distributions.

    Parameters
    ----------
    feats_src : (N_s, D) source features
    feats_tgt : (N_t, D) target features
    eps       : regularisation added to the covariance diagonal to avoid
                numerical issues with near-singular matrices.

    Returns
    -------
    float – Fréchet distance (non-negative)
    """
    mu_s = feats_src.mean(axis=0)
    mu_t = feats_tgt.mean(axis=0)

    sigma_s = np.cov(feats_src, rowvar=False) + eps * np.eye(feats_src.shape[1])
    sigma_t = np.cov(feats_tgt, rowvar=False) + eps * np.eye(feats_tgt.shape[1])

    # Matrix square-root of (Σ_s · Σ_t)
    product, _ = scipy.linalg.sqrtm(sigma_s @ sigma_t, disp=False)

    # Discard imaginary part produced by numerical noise
    if np.iscomplexobj(product):
        product = product.real

    diff_sq   = float(np.sum((mu_s - mu_t) ** 2))
    trace_sum = float(np.trace(sigma_s + sigma_t - 2.0 * product))

    fd = diff_sq + trace_sum
    # FD is theoretically non-negative; clamp numerical negatives
    return float(np.sqrt(max(fd, 0.0)))


class FrechetDistance(BaseMethod):
    """
    Fréchet Distance over penultimate-layer features.

    Large FD → distributions are far apart → accuracy gap likely larger.
    Hence a *positive* regression coefficient is expected for Δ Acc ~ FD.
    """

    name = "Fréchet"
    is_direct_estimator = False
    predicts_gap        = True

    def compute(
        self,
        source: Dict[str, np.ndarray],
        target: Dict[str, np.ndarray],
    ) -> float:
        """Return FD between source and target penultimate features."""
        return _frechet_distance(source["feats"], target["feats"])
