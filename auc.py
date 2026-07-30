"""
Discriminative Distance baselines.

The domain-adversarial intuition (Ben-David et al., 2006; Ganin et al., 2016)
suggests that a classifier that can distinguish source from target features
indicates a large distribution shift.

Two metrics are derived from this binary discriminator:

1. **Disc. AUC** – Area Under the ROC Curve of the discriminator on test split.
   A perfectly discriminable pair gives AUC = 1; indistinguishable pair gives
   AUC = 0.5.

2. **A-proxy** (Ben-David et al., 2006) – Eq. 11 in the paper:

       A-proxy = 2 (1 − 2 × error)

   where ``error`` is the test-set classification error of the discriminator.

The paper trains a *linear* discriminator (logistic regression) over the
penultimate features, matching the experimental setup in the paper
(Supplementary Sec. 10).  An MLP discriminator is also provided.

Empirical result: Despite good Pearson correlation with accuracy gaps, these
methods *overfit to synthetic shifts* and fail to transfer to natural shifts,
performing below the AC baseline.
"""

from __future__ import annotations

from typing import Dict, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def _run_discriminator(
    feats_src: np.ndarray,
    feats_tgt: np.ndarray,
    classifier: Literal["linear", "mlp"] = "linear",
    train_frac: float = 0.40,
    val_frac:   float = 0.10,
    seed: int   = 42,
) -> Dict[str, float]:
    """
    Train a binary discriminator to separate source from target features and
    compute AUC and A-proxy on a held-out test split.

    Split protocol (paper Sec. 10):
    * 40 % training, 10 % validation (hyper-param tuning), 50 % test.
    In this implementation the validation split is used for early stopping
    (MLP only) and the test split for final reporting.

    Parameters
    ----------
    feats_src   : (N_s, D) source features
    feats_tgt   : (N_t, D) target features
    classifier  : ``"linear"`` (default, matching paper) or ``"mlp"``
    train_frac  : fraction of each domain used for training
    val_frac    : fraction used for validation / model selection
    seed        : random seed for reproducibility

    Returns
    -------
    dict with keys ``auc``, ``error``, ``a_proxy``
    """
    rng = np.random.default_rng(seed)

    def split(feats):
        n      = len(feats)
        idx    = rng.permutation(n)
        n_tr   = int(n * train_frac)
        n_va   = int(n * val_frac)
        return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

    src_tr, src_va, src_te = split(feats_src)
    tgt_tr, tgt_va, tgt_te = split(feats_tgt)

    # Labels: 0 = source, 1 = target
    X_tr = np.vstack([feats_src[src_tr], feats_tgt[tgt_tr]])
    y_tr = np.hstack([np.zeros(len(src_tr)), np.ones(len(tgt_tr))])

    X_te = np.vstack([feats_src[src_te], feats_tgt[tgt_te]])
    y_te = np.hstack([np.zeros(len(src_te)), np.ones(len(tgt_te))])

    # Normalise features for stability
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)

    if classifier == "linear":
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        clf.fit(X_tr, y_tr)

    elif classifier == "mlp":
        # Light MLP (paper Supp.: 3 layers 512/256/128)
        clf = MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            max_iter=200,
            learning_rate_init=1e-4,
            alpha=1e-3,
            random_state=seed,
            early_stopping=True,
            validation_fraction=val_frac / (train_frac + val_frac),
            n_iter_no_change=10,
        )
        # Re-build combined tr+va for MLP with early stopping
        X_all = np.vstack([
            feats_src[np.hstack([src_tr, src_va])],
            feats_tgt[np.hstack([tgt_tr, tgt_va])],
        ])
        y_all = np.hstack([
            np.zeros(len(src_tr) + len(src_va)),
            np.ones(len(tgt_tr)  + len(tgt_va)),
        ])
        X_all = scaler.transform(X_all)   # already fitted above
        clf.fit(X_all, y_all)
    else:
        raise ValueError(f"Unknown classifier: {classifier!r}")

    proba      = clf.predict_proba(X_te)[:, 1]
    preds      = (proba >= 0.5).astype(int)

    auc    = float(roc_auc_score(y_te, proba))
    error  = float((preds != y_te).mean())
    a_proxy = float(2.0 * (1.0 - 2.0 * error))

    return {"auc": auc, "error": error, "a_proxy": a_proxy}


from method import BaseMethod


class DiscriminativeDistance(BaseMethod):
    """
    Discriminative Distance family (Disc. AUC, Disc. A-proxy).

    Trains a linear (or MLP) binary classifier on penultimate features
    to separate source from target, then uses AUC or A-proxy as the
    feature S for the downstream linear regressor.

    Parameters
    ----------
    metric     : ``"auc"`` (default) or ``"a_proxy"``
    classifier : ``"linear"`` (paper default) or ``"mlp"``
    """

    name = "Disc. AUC"
    is_direct_estimator = False
    predicts_gap        = True

    def __init__(
        self,
        metric:     Literal["auc", "a_proxy"] = "auc",
        classifier: Literal["linear", "mlp"]  = "linear",
    ) -> None:
        self.metric     = metric
        self.classifier = classifier
        self.name       = "Disc. AUC" if metric == "auc" else "Disc. A-proxy"

    def compute(
        self,
        source: Dict[str, np.ndarray],
        target: Dict[str, np.ndarray],
    ) -> float:
        """Train discriminator and return AUC or A-proxy."""
        results = _run_discriminator(
            source["feats"],
            target["feats"],
            classifier=self.classifier,
        )
        return results[self.metric]


class DiscriminativeAUC(DiscriminativeDistance):
    """Discriminative AUC shorthand."""
    def __init__(self, classifier: str = "linear") -> None:
        super().__init__(metric="auc", classifier=classifier)


class DiscriminativeAProxy(DiscriminativeDistance):
    """Discriminative A-proxy shorthand."""
    def __init__(self, classifier: str = "linear") -> None:
        super().__init__(metric="a_proxy", classifier=classifier)
