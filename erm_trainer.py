"""
Empirical Risk Minimisation (ERM) trainer.

Trains a FeatureExtractorModel on the pooled source domains using standard
cross-entropy loss.  This is the simplest domain-generalisation baseline and
the backbone for all similarity-based accuracy predictors in the paper.

Features
--------
* Mixed-precision training (torch.cuda.amp) when a GPU is available
* Cosine-annealing learning-rate schedule with linear warm-up
* Best-model checkpointing based on source-validation accuracy
* Structured logging to both console and a JSON-lines log file
* Reproducible training via seed fixing
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from model import FeatureExtractorModel

logger = logging.getLogger(__name__)


# ── Training utilities ────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


# ── Trainer ──────────────────────────────────────────────────────────────────

class ERMTrainer:
    """
    Trains a classifier with Empirical Risk Minimisation on pooled source domains.

    Parameters
    ----------
    model          : FeatureExtractorModel to train
    device         : torch device string
    lr             : initial learning rate
    weight_decay   : L2 regularisation
    optimizer_name : ``"sgd"`` | ``"adam"`` | ``"adamw"``
    momentum       : SGD momentum (ignored for Adam variants)
    use_amp        : enable automatic mixed precision (GPU only)
    seed           : random seed for reproducibility
    """

    def __init__(
        self,
        model:          FeatureExtractorModel,
        device:         str   = "cuda",
        lr:             float = 5e-4,
        weight_decay:   float = 1e-4,
        optimizer_name: str   = "sgd",
        momentum:       float = 0.9,
        use_amp:        bool  = True,
        seed:           int   = 42,
    ) -> None:
        self.model   = model.to(device)
        self.device  = device
        self.use_amp = use_amp and device.startswith("cuda")
        set_seed(seed)

        if optimizer_name == "sgd":
            self.optimizer = SGD(
                model.parameters(), lr=lr,
                momentum=momentum, weight_decay=weight_decay, nesterov=True
            )
        elif optimizer_name == "adam":
            self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name!r}")

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.scaler    = GradScaler() if self.use_amp else None

        self._history: List[Dict] = []

    # ── Public API ───────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader:   DataLoader,
        val_loaders:    Dict[str, DataLoader],
        n_epochs:       int   = 30,
        warmup_epochs:  int   = 5,
        patience:       int   = 10,
        save_dir:       str | Path | None = None,
        run_name:       str   = "erm",
    ) -> Dict:
        """
        Train for up to ``n_epochs`` epochs.

        Parameters
        ----------
        train_loader   : DataLoader over pooled source domains
        val_loaders    : dict of {domain_name: DataLoader} for validation
        n_epochs       : maximum number of epochs
        warmup_epochs  : number of linear warm-up epochs
        patience       : early-stopping patience (epochs without improvement)
        save_dir       : directory to save best model checkpoint and logs
        run_name       : identifier used in log filenames

        Returns
        -------
        dict with keys: best_epoch, best_val_acc, history
        """
        save_dir = Path(save_dir) if save_dir else None
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        # LR schedule: linear warm-up → cosine decay
        scheduler = self._build_scheduler(n_epochs, warmup_epochs)

        best_val_acc  = -1.0
        best_epoch    = 0
        epochs_no_imp = 0

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()

            # ── Training step ──────────────────────────────────────────────
            train_metrics = self._train_epoch(train_loader)
            scheduler.step()

            # ── Validation step ────────────────────────────────────────────
            val_metrics = {}
            for name, loader in val_loaders.items():
                val_metrics[name] = self._eval_epoch(loader)

            avg_val_acc = float(
                sum(v["accuracy"] for v in val_metrics.values()) / len(val_metrics)
            )

            elapsed = time.time() - t0
            lr_now  = self.optimizer.param_groups[0]["lr"]

            row = {
                "epoch":       epoch,
                "train_loss":  train_metrics["loss"],
                "train_acc":   train_metrics["accuracy"],
                "avg_val_acc": avg_val_acc,
                "lr":          lr_now,
                "time_s":      elapsed,
                **{f"val_{k}_acc": v["accuracy"] for k, v in val_metrics.items()},
            }
            self._history.append(row)

            logger.info(
                f"Epoch {epoch:3d}/{n_epochs}  "
                f"loss={train_metrics['loss']:.4f}  "
                f"train_acc={train_metrics['accuracy']:.4f}  "
                f"avg_val_acc={avg_val_acc:.4f}  "
                f"lr={lr_now:.2e}  [{elapsed:.1f}s]"
            )

            # ── Best-model checkpoint ──────────────────────────────────────
            if avg_val_acc > best_val_acc:
                best_val_acc  = avg_val_acc
                best_epoch    = epoch
                epochs_no_imp = 0
                if save_dir:
                    ckpt_path = save_dir / f"{run_name}_best.pt"
                    torch.save({
                        "epoch":        epoch,
                        "model_state":  self.model.state_dict(),
                        "optim_state":  self.optimizer.state_dict(),
                        "val_acc":      best_val_acc,
                    }, ckpt_path)
                    logger.info(f"  ✓  Saved best model → {ckpt_path}")
            else:
                epochs_no_imp += 1
                if epochs_no_imp >= patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement for {patience} epochs)."
                    )
                    break

        # Save full training history
        if save_dir:
            log_path = save_dir / f"{run_name}_history.jsonl"
            with open(log_path, "w") as f:
                for row in self._history:
                    f.write(json.dumps(row) + "\n")
            logger.info(f"Training history saved → {log_path}")

        return {
            "best_epoch":   best_epoch,
            "best_val_acc": best_val_acc,
            "history":      self._history,
        }

    # ── Private methods ───────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_corr = 0
        total_n    = 0

        for batch in loader:
            imgs   = batch[0].to(self.device, non_blocking=True)
            labels = batch[1].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with autocast():
                    logits, _ = self.model(imgs)
                    loss      = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, _ = self.model(imgs)
                loss      = self.criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

            total_loss += loss.item() * len(labels)
            total_corr += (logits.argmax(1) == labels).sum().item()
            total_n    += len(labels)

        return {
            "loss":     total_loss / total_n,
            "accuracy": total_corr / total_n,
        }

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_corr = 0
        total_n    = 0

        for batch in loader:
            imgs   = batch[0].to(self.device, non_blocking=True)
            labels = batch[1].to(self.device, non_blocking=True)

            logits, _ = self.model(imgs)
            loss      = self.criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            total_corr += (logits.argmax(1) == labels).sum().item()
            total_n    += len(labels)

        return {
            "loss":     total_loss / total_n,
            "accuracy": total_corr / total_n,
        }

    def _build_scheduler(self, n_epochs: int, warmup_epochs: int):
        warmup = LinearLR(
            self.optimizer,
            start_factor=1e-2,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, n_epochs - warmup_epochs),
            eta_min=1e-6,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

    def load_best(self, save_dir: str | Path, run_name: str = "erm") -> None:
        """Restore best checkpoint weights into the model."""
        ckpt_path = Path(save_dir) / f"{run_name}_best.pt"
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        logger.info(
            f"Loaded best model from epoch {ckpt['epoch']} "
            f"(val_acc={ckpt['val_acc']:.4f}) ← {ckpt_path}"
        )
