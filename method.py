"""
Abstract base class for all distributional-distance / confidence methods.

Every method must implement ``compute(source_dict, target_dict) → float``
which returns a *scalar* similarity/distance score S that can later be
used as input to a linear regressor to predict Δ Acc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class BaseMethod(ABC):
    """
    Interface for all similarity / distance measures.

    Subclasses set ``self.name`` and implement ``compute``.

    Methods that also directly estimate accuracy (AC, DoC-Feat) should set
    ``self.is_direct_estimator = True`` and implement ``predict_accuracy``.
    """

    #: Human-readable name shown in plots and tables
    name: str = "base"

    #: If True, this method produces a direct accuracy estimate (no regressor)
    is_direct_estimator: bool = False

    #: If True, the score S predicts Δ Acc (accuracy *gap*) rather than raw acc
    predicts_gap: bool = True

    def __call__(
        self,
        source: Dict[str, np.ndarray],
        target: Dict[str, np.ndarray],
    ) -> float:
        return self.compute(source, target)

    @abstractmethod
    def compute(
        self,
        source: Dict[str, np.ndarray],
        target: Dict[str, np.ndarray],
    ) -> float:
        """
        Compute the scalar similarity / distance between two domains.

        Parameters
        ----------
        source : dict with keys ``probs``, ``feats``, ``labels``, ``preds``,
                 ``accuracy``, ``confidence`` (output of
                 ``extract_features``).
        target : same structure as ``source``.

        Returns
        -------
        float – scalar score S
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
