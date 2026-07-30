from avg_conf           import AverageConfidence, AverageConfidenceTempScaled
from diff_conf          import DifferenceOfConfidences, DifferenceOfConfidencesFeat
from diff_ent          import DifferenceOfEntropy
from fid      import FrechetDistance
from mmd          import MaximumMeanDiscrepancy
from auc import (
    DiscriminativeDistance,
    DiscriminativeAUC,
    DiscriminativeAProxy,
)

ALL_METHODS = [
    AverageConfidence(),
    AverageConfidenceTempScaled(),
    DifferenceOfConfidences(),
    DifferenceOfConfidencesFeat(),
    DifferenceOfEntropy(),
    FrechetDistance(),
    MaximumMeanDiscrepancy(),
    DiscriminativeAUC(),
    DiscriminativeAProxy(),
]

__all__ = [
    "AverageConfidence",
    "AverageConfidenceTempScaled",
    "DifferenceOfConfidences",
    "DifferenceOfConfidencesFeat",
    "DifferenceOfEntropy",
    "FrechetDistance",
    "MaximumMeanDiscrepancy",
    "DiscriminativeDistance",
    "DiscriminativeAUC",
    "DiscriminativeAProxy",
    "ALL_METHODS",
]
