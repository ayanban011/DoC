"""
DoC-OOD: Predicting with Confidence on Unseen Distributions
=============================================================

Official implementation (adapted for domain generalisation) of:
    "Predicting with Confidence on Unseen Distributions"
    Guillory et al., ICCV 2021 — https://arxiv.org/abs/2107.03315

Benchmarks
----------
* PACS       (7 classes, 4 domains: Photo / Art / Cartoon / Sketch)
* OfficeHome (65 classes, 4 domains: Art / Clipart / Product / Real-World)
* DomainNet  (345 classes, 6 domains: Clipart / Infograph / Painting /
                                       Quickdraw / Real / Sketch)

Key ideas
---------
* **AC**   – Average Confidence: directly estimates target accuracy from mean
              max-softmax confidence. Biased but cheap.
* **DoC**  – Difference of Confidences: AC(source) − AC(target). Used as a
              feature for a linear regressor that predicts Δ accuracy.
* **DoE**  – Difference of (average) Entropy: entropy variant of DoC.
* Fréchet / MMD / Discriminative distances: comparison baselines.
"""

__version__ = "1.0.0"
__author__  = "Adapted from Guillory, Shankar, Ebrahimi, Darrell & Schmidt (2021)"
__license__ = "MIT"

from avg_conf           import AverageConfidence
from diff_conf          import DifferenceOfConfidences
from diff_ent          import DifferenceOfEntropy
from fid      import FrechetDistance
from mmd          import MaximumMeanDiscrepancy
from auc import DiscriminativeDistance

__all__ = [
    "AverageConfidence",
    "DifferenceOfConfidences",
    "DifferenceOfEntropy",
    "FrechetDistance",
    "MaximumMeanDiscrepancy",
    "DiscriminativeDistance",
]
