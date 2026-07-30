"""
Dataset wrappers for PACS, OfficeHome and DomainNet.

Expected directory layout
--------------------------
pacs/
    art_painting/   cartoon/   photo/   sketch/
        class_A/  class_B/ …

office_home/
    Art/   Clipart/   Product/   Real_World/
        class_A/  class_B/ …

domain_net/
    clipart/   infograph/   painting/   quickdraw/   real/   sketch/
        class_A/  class_B/ …

All datasets follow the ``ImageFolder`` convention so any sub-directory
structure with one folder per class is supported.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder

from transforms import get_eval_transform, get_train_transform


# ── Dataset meta-data ────────────────────────────────────────────────────────

DATASET_META: Dict[str, Dict] = {
    "pacs": {
        "domains":    ["art_painting", "cartoon", "photo", "sketch"],
        "num_classes": 7,
        "url":        "http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017",
    },
    "office_home": {
        "domains":    ["Art", "Clipart", "Product", "Real_World"],
        "num_classes": 65,
        "url":        "https://hemanthdv.github.io/officehome-dataset/",
    },
    "domain_net": {
        "domains":    ["clipart", "infograph", "painting",
                       "quickdraw", "real", "sketch"],
        "num_classes": 345,
        "url":        "http://ai.bu.edu/M3SDA/",
    },
}


# ── Core helpers ─────────────────────────────────────────────────────────────

class DomainDataset(Dataset):
    """
    Wraps an ``ImageFolder`` for one domain, augmenting each sample with
    a domain index so that multi-domain loaders stay compatible.
    """

    def __init__(
        self,
        root: str | Path,
        domain_name: str,
        domain_idx: int,
        transform: Optional[Callable] = None,
    ) -> None:
        self.domain_name = domain_name
        self.domain_idx  = domain_idx
        self.inner       = ImageFolder(str(root), transform=transform)
        self.classes     = self.inner.classes
        self.class_to_idx = self.inner.class_to_idx

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int):
        img, label = self.inner[idx]
        return img, label, self.domain_idx


class MultiDomainDataset(Dataset):
    """
    Concatenation of several ``DomainDataset`` instances with a unified
    class index.  Returns ``(image, class_label, domain_label)``.
    """

    def __init__(self, domain_datasets: Sequence[DomainDataset]) -> None:
        self.datasets     = list(domain_datasets)
        self.classes      = self.datasets[0].classes
        self._cumulative  = []
        running = 0
        for ds in self.datasets:
            running += len(ds)
            self._cumulative.append(running)

    def __len__(self) -> int:
        return self._cumulative[-1]

    def __getitem__(self, idx: int):
        for ds_idx, end in enumerate(self._cumulative):
            start = self._cumulative[ds_idx - 1] if ds_idx > 0 else 0
            if idx < end:
                return self.datasets[ds_idx][idx - start]
        raise IndexError(idx)


# ── Public factory ───────────────────────────────────────────────────────────

def build_domain_datasets(
    data_root: str | Path,
    dataset_name: str,
    target_domain: str,
    train_ratio: float = 0.8,
    img_size: int = 224,
    seed: int = 42,
) -> Tuple[MultiDomainDataset, Dict[str, Dataset], Dict[str, Dataset]]:
    """
    Build train and validation datasets for a leave-one-domain-out split.

    Parameters
    ----------
    data_root     : root of the dataset (contains one sub-folder per domain)
    dataset_name  : one of ``"pacs"``, ``"office_home"``, ``"domain_net"``
    target_domain : held-out domain for OOD evaluation
    train_ratio   : fraction of source-domain data used for training
    img_size      : spatial resolution fed to the model
    seed          : random seed for reproducible splits

    Returns
    -------
    train_dataset : ``MultiDomainDataset`` over source domains (training split)
    val_datasets  : dict domain→Dataset  (validation split of each source domain)
    target_dataset: Dataset for the held-out target domain (evaluation only)
    """
    rng    = random.Random(seed)
    meta   = DATASET_META[dataset_name]
    root   = Path(data_root)
    tr_tf  = get_train_transform(img_size)
    ev_tf  = get_eval_transform(img_size)

    source_domains = [d for d in meta["domains"] if d != target_domain]

    train_datasets : List[DomainDataset] = []
    val_datasets   : Dict[str, Dataset]  = {}

    for d_idx, domain in enumerate(source_domains):
        domain_root = root / domain
        if not domain_root.exists():
            raise FileNotFoundError(
                f"Domain folder not found: {domain_root}\n"
                f"Please download the dataset from {meta['url']}"
            )

        # Build the full dataset once to get indices then split
        full_ds = DomainDataset(domain_root, domain, d_idx, transform=ev_tf)
        n       = len(full_ds)
        indices = list(range(n))
        rng.shuffle(indices)
        split   = int(n * train_ratio)
        tr_idx, va_idx = indices[:split], indices[split:]

        # Training: with data-aug transforms
        tr_ds = DomainDataset(domain_root, domain, d_idx, transform=tr_tf)
        train_datasets.append(_SubsetDomain(tr_ds, tr_idx))

        # Validation: eval transforms
        val_datasets[domain] = Subset(full_ds, va_idx)

    # Target domain – evaluation only (no training labels used)
    all_domains    = meta["domains"]
    t_domain_idx   = all_domains.index(target_domain)
    target_root    = root / target_domain
    if not target_root.exists():
        raise FileNotFoundError(f"Target domain folder not found: {target_root}")
    target_dataset = DomainDataset(
        target_root, target_domain, t_domain_idx, transform=ev_tf
    )

    multi_train = MultiDomainDataset(train_datasets)

    return multi_train, val_datasets, target_dataset


class _SubsetDomain(Dataset):
    """Subset wrapper that preserves DomainDataset attributes."""

    def __init__(self, dataset: DomainDataset, indices: List[int]) -> None:
        self._ds      = dataset
        self._indices = indices
        self.domain_name  = dataset.domain_name
        self.domain_idx   = dataset.domain_idx
        self.classes      = dataset.classes
        self.class_to_idx = dataset.class_to_idx

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._ds[self._indices[idx]]


# ── Data-loader factory ──────────────────────────────────────────────────────

def build_dataloaders(
    train_dataset: Dataset,
    val_datasets : Dict[str, Dataset],
    target_dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, Dict[str, DataLoader], DataLoader]:
    """Return train / validation / target ``DataLoader`` objects."""

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loaders = {
        name: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        for name, ds in val_datasets.items()
    }

    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loaders, target_loader


# ── Calibration-set builder ──────────────────────────────────────────────────

def build_calibration_pairs(
    val_datasets: Dict[str, Dataset],
    batch_size: int = 64,
    num_workers: int = 4,
) -> List[Tuple[str, str, DataLoader, DataLoader]]:
    """
    Produce all pairwise (source_i, source_j) calibration combinations from the
    source-domain validation splits.  These pairs are used to fit the linear
    regression model R(S) → Δ Acc.

    Returns a list of tuples: (domain_a_name, domain_b_name, loader_a, loader_b)
    """
    names   = list(val_datasets.keys())
    pairs   = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            la = DataLoader(val_datasets[a], batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
            lb = DataLoader(val_datasets[b], batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
            pairs.append((a, b, la, lb))
    return pairs
