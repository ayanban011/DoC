"""
Batch feature / probability extraction from DataLoaders.

This module is the bridge between raw data and the similarity methods:
it runs the model in eval mode and collects, for every sample in a loader,
*  ``probs``  – softmax probabilities  (N, C)
*  ``feats``  – penultimate features   (N, D)
*  ``labels`` – ground-truth labels    (N,)
*  ``preds``  – argmax class           (N,)
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    temperature: float = 1.0,
    desc: str = "",
) -> Dict[str, np.ndarray]:
    """
    Run ``model`` over all batches in ``loader`` and collect outputs.

    Parameters
    ----------
    model       : A ``FeatureExtractorModel`` (or compatible) with signature
                  ``forward(x) → (logits, feats)``.
    loader      : DataLoader that yields ``(images, labels, …)``; only the
                  first two elements are used.
    device      : torch device string, e.g. ``"cuda"`` or ``"cpu"``.
    temperature : Softmax temperature (1.0 = standard, > 1 = softer).
    desc        : Progress-bar label.

    Returns
    -------
    dict with keys:
        ``probs``    – np.ndarray (N, C), float32
        ``feats``    – np.ndarray (N, D), float32
        ``labels``   – np.ndarray (N,),  int64
        ``preds``    – np.ndarray (N,),  int64
        ``accuracy`` – float, top-1 accuracy over the loader
        ``confidence``– float, mean max-softmax confidence
    """
    model.eval()
    model.to(device)

    all_probs:  list[torch.Tensor] = []
    all_feats:  list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in tqdm(loader, desc=desc or "Extracting features", leave=False):
        imgs   = batch[0].to(device, non_blocking=True)
        labels = batch[1]

        logits, feats = model(imgs)
        probs = torch.softmax(logits / temperature, dim=-1).cpu()

        all_probs.append(probs)
        all_feats.append(feats.cpu())
        all_labels.append(labels)

    probs_np  = torch.cat(all_probs).numpy().astype(np.float32)
    feats_np  = torch.cat(all_feats).numpy().astype(np.float32)
    labels_np = torch.cat(all_labels).numpy().astype(np.int64)
    preds_np  = probs_np.argmax(axis=1).astype(np.int64)

    accuracy   = float((preds_np == labels_np).mean())
    confidence = float(probs_np.max(axis=1).mean())

    return {
        "probs":      probs_np,
        "feats":      feats_np,
        "labels":     labels_np,
        "preds":      preds_np,
        "accuracy":   accuracy,
        "confidence": confidence,
    }


def extract_all_domain_features(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    device: str,
    temperature: float = 1.0,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract features for every domain in ``loaders``.

    Returns
    -------
    dict mapping domain_name → feature-dict (same format as
    ``extract_features`` output).
    """
    results = {}
    for name, loader in loaders.items():
        results[name] = extract_features(
            model, loader, device,
            temperature=temperature,
            desc=f"  [{name}]",
        )
    return results


def get_shared_class_mask(
    probs_source: np.ndarray,
    probs_target: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return boolean masks selecting only class columns that are present in
    both source and target softmax outputs (non-zero probability mass).

    In the multi-domain DG setting, source and target always share the same
    label space, so this returns the full column range.  The function exists
    for compatibility with the paper's formulation where some shifted
    datasets have a *subset* of labels.
    """
    n_classes_src = probs_source.shape[1]
    n_classes_tgt = probs_target.shape[1]
    assert n_classes_src == n_classes_tgt, (
        "Source and target must have the same number of classes in the DG setting."
    )
    # All classes are shared
    mask = np.ones(n_classes_src, dtype=bool)
    return mask, mask
