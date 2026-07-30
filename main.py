#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from init_data import (
    DATASET_META,
    build_domain_datasets,
    build_dataloaders,
    build_calibration_pairs,
)
from transforms import get_corruption_transforms
from model import build_model, TemperatureScaledModel
from model_utils import extract_features, extract_all_domain_features
from init_method import (
    AverageConfidence,
    AverageConfidenceTempScaled,
    DifferenceOfConfidences,
    DifferenceOfConfidencesFeat,
    DifferenceOfEntropy,
    FrechetDistance,
    MaximumMeanDiscrepancy,
    DiscriminativeAUC,
    DiscriminativeAProxy,
)
from evaluation import summarise_results, relative_improvement
from regression import AccuracyPredictor
from erm_trainer import ERMTrainer, set_seed
from report import ExperimentReport
from viz import (
    plot_mae_bar_chart,
    plot_predicted_vs_actual,
    plot_confidence_histogram,
    plot_doc_vs_delta_acc,
    plot_training_curves,
    plot_heatmap,
    plot_relative_improvements,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_experiment")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DoC-OOD: Predicting accuracy on unseen domains.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Dataset ──────────────────────────────────────────────────────────────
    p.add_argument("--dataset",      type=str, required=True,
                   choices=["pacs", "office_home", "domain_net"],
                   help="Which dataset to run on.")
    p.add_argument("--data_root",    type=str, required=True,
                   help="Root directory of the dataset (one sub-folder per domain).")
    p.add_argument("--target_domain", type=str, default=None,
                   help="Specific target domain to evaluate. Default: loop over all.")

    # ── Model ────────────────────────────────────────────────────────────────
    p.add_argument("--backbone",     type=str, default="resnet50",
                   choices=["resnet18", "resnet34", "resnet50",
                             "resnet101", "resnet152",
                             "vgg16", "vgg19",
                             "densenet121", "densenet169"],
                   help="Encoder architecture.")
    p.add_argument("--pretrained",   action="store_true", default=True,
                   help="Use ImageNet-pretrained weights.")
    p.add_argument("--checkpoint",   type=str, default=None,
                   help="Path to a saved model checkpoint (.pt) to load instead of training.")

    # ── Training ─────────────────────────────────────────────────────────────
    p.add_argument("--skip_train",   action="store_true",
                   help="Skip training (requires --checkpoint).")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--warmup",       type=int,   default=5,
                   help="Number of linear warm-up epochs.")
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--optimizer",    type=str,   default="sgd",
                   choices=["sgd", "adam", "adamw"])
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--patience",     type=int,   default=10,
                   help="Early-stopping patience (epochs).")
    p.add_argument("--train_ratio",  type=float, default=0.80,
                   help="Fraction of source data used for training (rest = validation).")

    # ── Regression ────────────────────────────────────────────────────────────
    p.add_argument("--reg_model",    type=str,   default="linear",
                   choices=["linear", "ridge", "mlp"],
                   help="Regression model type for calibrating DoC/DoE/etc.")
    p.add_argument("--temperature_scaling", action="store_true",
                   help="Apply temperature scaling before computing AC.")

    # ── Misc ─────────────────────────────────────────────────────────────────
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--device",       type=str,   default=None,
                   help="cuda / cpu (auto-detected if not set).")
    p.add_argument("--output_dir",   type=str,   default="results",
                   help="Root directory for all outputs (checkpoints, plots, CSVs).")
    p.add_argument("--no_plots",     action="store_true",
                   help="Disable matplotlib figure generation.")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# CORE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_single_target(
    args:          argparse.Namespace,
    target_domain: str,
    device:        str,
    report:        ExperimentReport,
) -> None:
    """
    Full pipeline for one leave-one-domain-out split:

    1. Build datasets (source train/val + target)
    2. Train ERM model on source domains  (or load checkpoint)
    3. (Optional) Temperature-scale the model on source val
    4. Extract features + probabilities for every domain
    5. Build calibration pairs from source validation splits
    6. For each method:
         a. Compute similarity score S on each calibration pair
         b. Fit linear regressor R(S) → ΔAcc  [for regression-based methods]
         c. Compute S on source→target pair
         d. Predict target accuracy
    7. Evaluate predictions, update report, save artefacts
    """
    meta       = DATASET_META[args.dataset]
    out_dir    = Path(args.output_dir) / args.dataset / target_domain
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    logger.info("=" * 70)
    logger.info(f"  Dataset : {args.dataset.upper()}  |  Target : {target_domain}")
    logger.info(f"  Backbone: {args.backbone}           |  Device : {device}")
    logger.info("=" * 70)

    # ── 1. Datasets ───────────────────────────────────────────────────────────
    logger.info("[1/7] Building datasets …")
    train_ds, val_dss, target_ds = build_domain_datasets(
        data_root     = args.data_root,
        dataset_name  = args.dataset,
        target_domain = target_domain,
        train_ratio   = args.train_ratio,
        seed          = args.seed,
    )

    train_loader, val_loaders, target_loader = build_dataloaders(
        train_ds, val_dss, target_ds,
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
    )

    source_domains = list(val_dss.keys())
    logger.info(f"  Source domains : {source_domains}")
    logger.info(f"  Target domain  : {target_domain}")
    logger.info(f"  Train samples  : {len(train_ds):,}")
    logger.info(f"  Target samples : {len(target_ds):,}")

    # ── 2. Model ──────────────────────────────────────────────────────────────
    logger.info("[2/7] Building model …")
    model = build_model(
        backbone    = args.backbone,
        num_classes = meta["num_classes"],
        pretrained  = args.pretrained,
        device      = device,
    )

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state)
        logger.info(f"  Loaded checkpoint: {args.checkpoint}")

    elif not args.skip_train:
        logger.info("[2/7] Training ERM model …")
        trainer = ERMTrainer(
            model          = model,
            device         = device,
            lr             = args.lr,
            weight_decay   = args.weight_decay,
            optimizer_name = args.optimizer,
            seed           = args.seed,
        )
        train_result = trainer.fit(
            train_loader  = train_loader,
            val_loaders   = val_loaders,
            n_epochs      = args.epochs,
            warmup_epochs = args.warmup,
            patience      = args.patience,
            save_dir      = out_dir,
            run_name      = "erm",
        )
        trainer.load_best(out_dir, "erm")

        if not args.no_plots:
            plot_training_curves(
                train_result["history"],
                save_path=out_dir / "training_curves.png",
            )
        logger.info(
            f"  Best val acc: {train_result['best_val_acc']:.4f} "
            f"at epoch {train_result['best_epoch']}"
        )
    else:
        logger.warning("--skip_train set but no --checkpoint provided. "
                       "Using randomly initialised model.")

    model.eval()

    # ── 3. Temperature scaling (optional) ────────────────────────────────────
    if args.temperature_scaling:
        logger.info("[3/7] Calibrating model with temperature scaling …")
        ts_model = TemperatureScaledModel(model)
        # Calibrate on combined source validation set
        all_val_loaders = list(val_loaders.values())
        # Use the first source domain val loader for calibration
        T_opt = ts_model.calibrate(all_val_loaders[0], device=device)
        logger.info(f"  Optimal temperature T = {T_opt:.4f}")
        ts_temperature = T_opt
    else:
        ts_temperature = 1.0

    # ── 4. Feature extraction ─────────────────────────────────────────────────
    logger.info("[4/7] Extracting features …")

    # Source validation domains
    val_features: Dict[str, Dict] = {}
    for name, loader in val_loaders.items():
        logger.info(f"  Extracting: {name} (val)")
        val_features[name] = extract_features(
            model, loader, device,
            temperature=ts_temperature,
            desc=f"  {name}",
        )
        logger.info(
            f"    acc={val_features[name]['accuracy']:.4f}  "
            f"conf={val_features[name]['confidence']:.4f}  "
            f"n={len(val_features[name]['labels'])}"
        )

    # Pooled source: concat all val features (represents distribution B)
    pooled_probs = np.concatenate([v["probs"] for v in val_features.values()])
    pooled_feats = np.concatenate([v["feats"] for v in val_features.values()])
    pooled_lbls  = np.concatenate([v["labels"] for v in val_features.values()])
    pooled_preds = np.concatenate([v["preds"]  for v in val_features.values()])
    pooled_acc   = float((pooled_preds == pooled_lbls).mean())

    source_dict = {
        "probs":      pooled_probs,
        "feats":      pooled_feats,
        "labels":     pooled_lbls,
        "preds":      pooled_preds,
        "accuracy":   pooled_acc,
        "confidence": float(pooled_probs.max(axis=1).mean()),
    }
    logger.info(f"  Pooled source  acc={pooled_acc:.4f}  conf={source_dict['confidence']:.4f}")

    # Target domain
    logger.info(f"  Extracting: {target_domain} (target)")
    target_dict = extract_features(
        model, target_loader, device,
        temperature=ts_temperature,
        desc=f"  {target_domain}",
    )
    actual_target_acc = target_dict["accuracy"]
    logger.info(
        f"  Target acc={actual_target_acc:.4f}  "
        f"conf={target_dict['confidence']:.4f}  "
        f"n={len(target_dict['labels'])}"
    )

    # ── 5. Calibration pairs ──────────────────────────────────────────────────
    logger.info("[5/7] Building calibration set …")

    # Use every pairwise combination of source-domain val sets as calibration shifts.
    # Also augment with synthetic corruptions for a richer calibration set.
    cal_pairs: List[Dict] = []

    domain_names = list(val_features.keys())
    for i, dn_a in enumerate(domain_names):
        for j, dn_b in enumerate(domain_names):
            if i >= j:
                continue
            feat_a = val_features[dn_a]
            feat_b = val_features[dn_b]

            delta_acc_ab = feat_a["accuracy"] - feat_b["accuracy"]
            delta_acc_ba = feat_b["accuracy"] - feat_a["accuracy"]

            cal_pairs.append({
                "base": feat_a, "target": feat_b, "delta": delta_acc_ab,
                "label": f"{dn_a}→{dn_b}",
            })
            cal_pairs.append({
                "base": feat_b, "target": feat_a, "delta": delta_acc_ba,
                "label": f"{dn_b}→{dn_a}",
            })

    # Synthetic calibration: apply lightweight corruptions to source val data
    # and use the corrupted features as additional calibration targets.
    logger.info(f"  Adding synthetic corruption calibration points …")
    from torch.utils.data import DataLoader, Subset
    from transforms import get_corruption_transforms

    corr_transforms = get_corruption_transforms(severity=3)
    for corr_name, corr_tf in corr_transforms.items():
        for dom_name, dom_ds in val_dss.items():
            # Re-wrap dataset with corruption transform
            corr_ds   = _CorruptedDataset(dom_ds, corr_tf)
            corr_ldr  = DataLoader(corr_ds, batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.num_workers)
            corr_feat = extract_features(
                model, corr_ldr, device,
                temperature=ts_temperature,
                desc=f"  [{corr_name}/{dom_name}]",
            )
            base_feat = val_features[dom_name]
            delta     = base_feat["accuracy"] - corr_feat["accuracy"]
            cal_pairs.append({
                "base": base_feat, "target": corr_feat, "delta": delta,
                "label": f"{dom_name}+{corr_name}",
            })

    logger.info(f"  Total calibration pairs: {len(cal_pairs)}")

    # ── 6. Run all methods ────────────────────────────────────────────────────
    logger.info("[6/7] Computing similarity measures and predicting accuracy …")

    # Define methods
    regression_methods = [
        DifferenceOfConfidences(),
        DifferenceOfEntropy(),
        FrechetDistance(),
        MaximumMeanDiscrepancy(),
        DiscriminativeAUC(),
        DiscriminativeAProxy(),
    ]
    direct_methods = [
        AverageConfidence(),
        DifferenceOfConfidencesFeat(),
    ]
    if args.temperature_scaling:
        direct_methods.append(AverageConfidenceTempScaled())

    results: Dict[str, Dict] = {}

    # ── Direct estimators (no regressor needed) ────────────────────────────
    for method in direct_methods:
        pred_acc = method.predict_accuracy(source_dict, target_dict) \
                   if hasattr(method, "predict_accuracy") \
                   else method.compute(source_dict, target_dict)
        err = abs(pred_acc - actual_target_acc)
        results[method.name] = {
            "predicted_acc": pred_acc,
            "actual_acc":    actual_target_acc,
            "abs_error":     err,
            "score":         None,
        }
        logger.info(
            f"  {method.name:<20} pred={pred_acc:.4f}  "
            f"actual={actual_target_acc:.4f}  |err|={err:.4f}"
        )

    # ── Regression-based methods ───────────────────────────────────────────
    for method in regression_methods:
        # Build calibration (score, delta_acc) pairs
        cal_scores = []
        cal_deltas = []
        for pair in cal_pairs:
            try:
                s = method.compute(pair["base"], pair["target"])
                cal_scores.append(s)
                cal_deltas.append(pair["delta"])
            except Exception as exc:
                logger.debug(f"  [{method.name}] skipped pair '{pair['label']}': {exc}")

        if len(cal_scores) < 2:
            logger.warning(f"  [{method.name}] insufficient calibration pairs, skipping.")
            continue

        # Fit regressor
        predictor = AccuracyPredictor(model_type=args.reg_model)
        predictor.fit(cal_scores, cal_deltas)

        # Predict on source→target
        target_score = method.compute(source_dict, target_dict)
        pred_delta   = predictor.predict_gap(target_score)
        pred_acc     = pooled_acc - pred_delta
        err          = abs(pred_acc - actual_target_acc)

        results[method.name] = {
            "predicted_acc":  pred_acc,
            "actual_acc":     actual_target_acc,
            "abs_error":      err,
            "score":          target_score,
            "predicted_delta": pred_delta,
            "actual_delta":   pooled_acc - actual_target_acc,
            "cal_scores":     cal_scores,
            "cal_deltas":     cal_deltas,
        }

        # Save regressor
        predictor.save(out_dir / f"regressor_{method.name.replace(' ', '_')}.pkl")

        logger.info(
            f"  {method.name:<20} score={target_score:+.4f}  "
            f"Δpred={pred_delta:+.4f}  pred={pred_acc:.4f}  "
            f"actual={actual_target_acc:.4f}  |err|={err:.4f}"
        )

    # Base accuracy (naive baseline: assume target acc = source acc)
    base_err = abs(pooled_acc - actual_target_acc)
    results["BaseAcc"] = {
        "predicted_acc": pooled_acc,
        "actual_acc":    actual_target_acc,
        "abs_error":     base_err,
        "score":         None,
    }
    logger.info(
        f"  {'BaseAcc':<20} pred={pooled_acc:.4f}  "
        f"actual={actual_target_acc:.4f}  |err|={base_err:.4f}"
    )

    # ── 7. Artefacts ─────────────────────────────────────────────────────────
    logger.info("[7/7] Saving artefacts …")

    # Save raw results
    summary_path = out_dir / "raw_results.json"
    safe_results = {
        k: {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
            for kk, vv in v.items()
            if not isinstance(vv, (list, np.ndarray)) or kk not in ("cal_scores", "cal_deltas")}
        for k, v in results.items()
    }
    with open(summary_path, "w") as f:
        json.dump(safe_results, f, indent=2)

    # Update report
    for method_name, res in results.items():
        report.add_result(
            method        = method_name,
            target_domain = target_domain,
            predicted_acc = res["predicted_acc"],
            actual_acc    = res["actual_acc"],
            score         = res.get("score"),
        )

    # Plots
    if not args.no_plots:
        # Confidence histograms
        conf_dict = {name: v["probs"] for name, v in val_features.items()}
        conf_dict[target_domain] = target_dict["probs"]
        plot_confidence_histogram(
            conf_dict,
            save_path=out_dir / "confidence_histograms.png",
        )

        # DoC feature vs ΔAcc (calibration scatter)
        doc_method = DifferenceOfConfidences()
        if "DoC" in results and results["DoC"].get("cal_scores"):
            doc_res = results["DoC"]
            plot_doc_vs_delta_acc(
                doc_scores   = doc_res["cal_scores"],
                delta_accs   = doc_res["cal_deltas"],
                save_path    = out_dir / "doc_vs_delta_acc.png",
            )

        # Predicted vs actual per method
        method_names = list(results.keys())
        pred_accs    = [results[m]["predicted_acc"] for m in method_names]
        act_accs     = [results[m]["actual_acc"]    for m in method_names]
        mae_vals     = [results[m]["abs_error"]      for m in method_names]

        plot_mae_bar_chart(
            method_names = method_names,
            mae_values   = mae_vals,
            title        = f"MAE per Method — {args.dataset.upper()} → {target_domain}",
            save_path    = out_dir / "mae_bar.png",
        )

    logger.info(f"  Outputs → {out_dir.resolve()}")
    logger.info("")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: corrupted dataset wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class _CorruptedDataset(torch.utils.data.Dataset):
    """
    Wrap an existing dataset and apply a corruption transform to the original
    PIL image.

    Correctly handles:
      - torch.utils.data.Subset
      - custom datasets wrapping another dataset via `_ds`
      - torchvision.datasets.ImageFolder
      - nested Subset objects
    """

    def __init__(self, dataset, corrupt_transform):
        self._ds = dataset
        self._transform = corrupt_transform

    def __len__(self):
        return len(self._ds)

    def _get_original_dataset_and_index(self, idx):
        """
        Recursively unwrap Subset and custom dataset wrappers until reaching
        the underlying dataset that contains `.samples`.

        Returns:
            base_dataset: dataset with `.samples`
            original_idx: index into base_dataset.samples
        """

        dataset = self._ds
        current_idx = idx

        while True:

            # ---------------------------------------------------------
            # Case 1: torch.utils.data.Subset
            # ---------------------------------------------------------
            if isinstance(dataset, torch.utils.data.Subset):
                current_idx = dataset.indices[current_idx]
                dataset = dataset.dataset
                continue

            # ---------------------------------------------------------
            # Case 2: Custom wrapper using `_ds`
            # ---------------------------------------------------------
            if hasattr(dataset, "_ds"):
                dataset = dataset._ds
                continue

            # ---------------------------------------------------------
            # Case 3: Custom wrapper using `inner`
            # ---------------------------------------------------------
            if hasattr(dataset, "inner"):
                dataset = dataset.inner
                continue

            # ---------------------------------------------------------
            # We reached the underlying dataset
            # ---------------------------------------------------------
            break

        return dataset, current_idx

    def __getitem__(self, idx):

        from PIL import Image

        # Get underlying dataset and correct original index
        base_dataset, original_idx = \
            self._get_original_dataset_and_index(idx)

        # -------------------------------------------------------------
        # torchvision ImageFolder
        # -------------------------------------------------------------
        if hasattr(base_dataset, "samples"):

            path, label = base_dataset.samples[original_idx]

            # Load original PIL image
            pil_img = Image.open(path).convert("RGB")

        else:
            # ---------------------------------------------------------
            # Fallback for datasets that don't expose `.samples`
            # ---------------------------------------------------------
            item = base_dataset[original_idx]

            if isinstance(item, (tuple, list)):
                pil_img, label = item[0], item[1]
            else:
                raise TypeError(
                    f"Unsupported dataset type: {type(base_dataset)}. "
                    "Dataset does not expose `.samples`."
                )

            # If the underlying dataset returns a tensor,
            # convert it back to PIL.
            if torch.is_tensor(pil_img):
                from torchvision.transforms import ToPILImage
                pil_img = ToPILImage()(pil_img)

        # -------------------------------------------------------------
        # Apply corruption transform
        # -------------------------------------------------------------
        tensor = self._transform(pil_img)

        return tensor, label

# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_and_plot(report: ExperimentReport, output_dir: Path, no_plots: bool) -> None:
    """Print summary tables and generate aggregate plots."""
    logger.info("=" * 70)
    logger.info("  AGGREGATE RESULTS")
    logger.info("=" * 70)

    report.print_summary()
    report.print_per_domain()
    report.save(output_dir)

    if no_plots:
        return

    # Gather for aggregate MAE bar chart
    summ        = report.summary_by_method()
    m_names     = list(summ.keys())
    m_maes      = [summ[m]["mae_mean"] for m in m_names]
    m_stds      = [summ[m]["mae_std"]  for m in m_names]

    plot_mae_bar_chart(
        method_names = m_names,
        mae_values   = m_maes,
        std_values   = m_stds,
        title        = f"Aggregate MAE — {report.dataset.upper()}",
        save_path    = output_dir / "aggregate_mae_bar.png",
    )

    # Relative improvement over AC baseline
    ac_mae = summ.get("AC", {}).get("mae_mean", None)
    if ac_mae is not None:
        rel_impr = [relative_improvement(ac_mae, summ[m]["mae_mean"]) for m in m_names]
        plot_relative_improvements(
            method_names  = m_names,
            rel_impr      = rel_impr,
            baseline_name = "AC",
            save_path     = output_dir / "relative_improvement.png",
        )

    # MAE heat map (methods × target domains)
    rows_by_method  = list(summ.keys())
    target_domains  = sorted({r["target_domain"] for r in report._rows})
    mae_matrix      = np.full((len(rows_by_method), len(target_domains)), np.nan)

    lookup = {}
    for row in report._rows:
        lookup[(row["method"], row["target_domain"])] = row["abs_error"]

    for i, m in enumerate(rows_by_method):
        for j, d in enumerate(target_domains):
            val = lookup.get((m, d), np.nan)
            mae_matrix[i, j] = val

    plot_heatmap(
        method_names = rows_by_method,
        domain_names = target_domains,
        mae_matrix   = mae_matrix,
        title        = f"MAE Heatmap — {report.dataset.upper()}",
        save_path    = output_dir / "mae_heatmap.png",
    )

    logger.info(f"Aggregate plots → {output_dir.resolve()}")

    # Print LaTeX table
    print("\n" + "─" * 60)
    print("LaTeX table:")
    print("─" * 60)
    print(report.to_latex())
    print("─" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args   = parse_args()
    t_start = time.time()

    # Device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Validate dataset
    meta = DATASET_META.get(args.dataset)
    if meta is None:
        logger.error(f"Unknown dataset '{args.dataset}'.")
        sys.exit(1)

    all_domains    = meta["domains"]
    target_domains = [args.target_domain] if args.target_domain else all_domains

    # Validate target domains
    for td in target_domains:
        if td not in all_domains:
            logger.error(f"Domain '{td}' not in {args.dataset}: {all_domains}")
            sys.exit(1)

    output_root = Path(args.output_dir) / args.dataset
    output_root.mkdir(parents=True, exist_ok=True)

    report = ExperimentReport(dataset=args.dataset, backbone=args.backbone)

    # ── Run leave-one-domain-out ──────────────────────────────────────────────
    for i, target_domain in enumerate(target_domains):
        logger.info(
            f"\n[Target {i+1}/{len(target_domains)}] "
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        try:
            run_single_target(args, target_domain, device, report)
        except FileNotFoundError as exc:
            logger.error(str(exc))
            logger.error("Have you downloaded the dataset? See README.md for instructions.")
            sys.exit(1)
        except Exception as exc:
            logger.exception(f"Error processing target domain '{target_domain}': {exc}")
            raise

    # ── Aggregate ─────────────────────────────────────────────────────────────
    aggregate_and_plot(report, output_root, args.no_plots)

    elapsed = time.time() - t_start
    logger.info(f"\nTotal runtime: {elapsed/60:.1f} min")
    logger.info(f"Results saved to: {output_root.resolve()}")


if __name__ == "__main__":
    main()
