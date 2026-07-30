"""
Visualisation utilities.

Reproduces the key figures from the paper adapted to the domain-generalisation
setting:

* ``plot_predicted_vs_actual``     – scatter plot (x=predicted, y=actual acc)
* ``plot_mae_bar_chart``           – horizontal bar chart of MAE per method
* ``plot_confidence_histogram``    – AC confidence histogram per domain
* ``plot_doc_vs_delta_acc``        – DoC feature vs Δ Acc scatter
* ``plot_training_curves``         – loss / accuracy over epochs
* ``plot_heatmap``                 – per-domain × per-method MAE heat map
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")            # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ── Style defaults ───────────────────────────────────────────────────────────

PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#1b9e77", "#d95f02",
]
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        120,
})


# ── Helper ───────────────────────────────────────────────────────────────────

def _save_or_show(fig: plt.Figure, path: Optional[str | Path]) -> None:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ── Figures ──────────────────────────────────────────────────────────────────

def plot_predicted_vs_actual(
    actual:       Sequence[float],
    predicted:    Sequence[float],
    method_name:  str,
    domain_names: Optional[Sequence[str]] = None,
    mae_val:      Optional[float] = None,
    r2_val:       Optional[float] = None,
    save_path:    Optional[str | Path] = None,
) -> plt.Figure:
    """
    Scatter plot of predicted vs actual accuracy on target domains.

    A perfect predictor lies on the diagonal y = x (shown as dashed line).
    """
    actual    = np.array(actual)
    predicted = np.array(predicted)

    fig, ax = plt.subplots(figsize=(5, 5))

    # Diagonal line (perfect prediction)
    lo = min(actual.min(), predicted.min()) - 0.02
    hi = max(actual.max(), predicted.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.6, label="Perfect")

    # Best fit line
    coef = np.polyfit(predicted, actual, 1)
    x_fit = np.linspace(predicted.min(), predicted.max(), 100)
    ax.plot(x_fit, np.polyval(coef, x_fit), color=PALETTE[0],
            linewidth=1.5, alpha=0.7, label="Best fit")

    # Data points
    for i, (x, y) in enumerate(zip(predicted, actual)):
        color = PALETTE[i % len(PALETTE)]
        label = domain_names[i] if domain_names else None
        ax.scatter(x, y, color=color, s=60, zorder=5, label=label)

    # Annotation
    ann_parts = []
    if mae_val is not None:
        ann_parts.append(f"MAE={mae_val:.3f}")
    if r2_val is not None:
        ann_parts.append(f"R²={r2_val:.3f}")
    if ann_parts:
        ax.text(0.05, 0.95, "  ".join(ann_parts),
                transform=ax.transAxes, fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax.set_xlabel("Predicted Accuracy")
    ax.set_ylabel("Actual Accuracy")
    ax.set_title(f"Predicted vs Actual — {method_name}")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    if domain_names:
        ax.legend(fontsize=8, loc="lower right")

    _save_or_show(fig, save_path)
    return fig


def plot_mae_bar_chart(
    method_names: Sequence[str],
    mae_values:   Sequence[float],
    std_values:   Optional[Sequence[float]] = None,
    highlight:    Optional[str] = "DoC",
    title:        str = "MAE by Method",
    save_path:    Optional[str | Path] = None,
) -> plt.Figure:
    """
    Horizontal bar chart comparing MAE across methods.
    The best-performing method (DoC) is highlighted.
    """
    n      = len(method_names)
    y_pos  = np.arange(n)
    colors = [
        PALETTE[0] if (highlight and highlight in m) else PALETTE[7]
        for m in method_names
    ]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * n)))

    bars = ax.barh(y_pos, mae_values, color=colors, xerr=std_values,
                   capsize=4, error_kw={"elinewidth": 1.5})

    # Value labels
    for bar, val in zip(bars, mae_values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(method_names)
    ax.set_xlabel("Mean Absolute Error (MAE) ↓")
    ax.set_title(title)
    ax.invert_yaxis()

    _save_or_show(fig, save_path)
    return fig


def plot_confidence_histogram(
    domain_features: Dict[str, np.ndarray],
    save_path:       Optional[str | Path] = None,
) -> plt.Figure:
    """
    Confidence histogram per domain (Figure 1 left in the paper).

    Parameters
    ----------
    domain_features : dict mapping domain name → probs array (N, C)
    """
    n_domains = len(domain_features)
    fig, axes = plt.subplots(1, n_domains, figsize=(4 * n_domains, 3.5), sharey=True)
    if n_domains == 1:
        axes = [axes]

    for ax, (name, probs) in zip(axes, domain_features.items()):
        confidences = probs.max(axis=1)
        ac          = confidences.mean()
        ax.hist(confidences, bins=30, color=PALETTE[list(domain_features).index(name)],
                alpha=0.75, edgecolor="white", linewidth=0.3)
        ax.axvline(ac, color="black", linestyle="--", linewidth=1.5, label=f"AC={ac:.3f}")
        ax.set_title(name)
        ax.set_xlabel("Max Confidence")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Count")
    fig.suptitle("Confidence Distributions per Domain", y=1.01)
    fig.tight_layout()

    _save_or_show(fig, save_path)
    return fig


def plot_doc_vs_delta_acc(
    doc_scores:  Sequence[float],
    delta_accs:  Sequence[float],
    domain_pairs: Optional[Sequence[str]] = None,
    pearson_r:   Optional[float] = None,
    save_path:   Optional[str | Path] = None,
) -> plt.Figure:
    """
    Scatter of DoC feature vs actual Δ Acc with regression line.
    Reproduces Figure 13 (confidence-based panel) of the paper.
    """
    x = np.array(doc_scores)
    y = np.array(delta_accs)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    coef   = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, np.polyval(coef, x_line),
            color=PALETTE[0], linewidth=2, alpha=0.8, label="Best fit")

    for i, (xi, yi) in enumerate(zip(x, y)):
        label = domain_pairs[i] if domain_pairs else None
        ax.scatter(xi, yi, color=PALETTE[i % len(PALETTE)], s=60,
                   zorder=5, label=label)

    if pearson_r is not None:
        ax.text(0.05, 0.95, f"ρ = {pearson_r:.3f}",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax.set_xlabel("DoC(source, target)")
    ax.set_ylabel("Δ Accuracy  (source − target)")
    ax.set_title("DoC Feature vs Accuracy Gap")
    ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    ax.axvline(0, color="grey", linestyle=":", linewidth=0.8)
    if domain_pairs:
        ax.legend(fontsize=8, loc="lower right")

    _save_or_show(fig, save_path)
    return fig


def plot_training_curves(
    history:   List[Dict],
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Loss and accuracy training curves."""
    epochs   = [r["epoch"]       for r in history]
    tr_loss  = [r["train_loss"]  for r in history]
    tr_acc   = [r["train_acc"]   for r in history]
    val_acc  = [r["avg_val_acc"] for r in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, tr_loss, color=PALETTE[0], label="Train loss")
    ax1.set_xlabel("Epoch");  ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("Training Loss");  ax1.legend()

    ax2.plot(epochs, tr_acc,  color=PALETTE[0], label="Train acc")
    ax2.plot(epochs, val_acc, color=PALETTE[1], linestyle="--", label="Val acc (avg)")
    ax2.set_xlabel("Epoch");  ax2.set_ylabel("Top-1 Accuracy")
    ax2.set_title("Accuracy Curves");  ax2.legend()

    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_heatmap(
    method_names: Sequence[str],
    domain_names: Sequence[str],
    mae_matrix:   np.ndarray,
    title:        str = "MAE Heatmap (methods × domains)",
    save_path:    Optional[str | Path] = None,
) -> plt.Figure:
    """
    Heat map of MAE for each (method, target domain) combination.

    Parameters
    ----------
    mae_matrix : (n_methods, n_domains) array of MAE values
    """
    fig, ax = plt.subplots(figsize=(max(5, 1.5 * len(domain_names)),
                                    max(4, 0.6 * len(method_names))))

    im = ax.imshow(mae_matrix, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=mae_matrix.max())
    plt.colorbar(im, ax=ax, label="MAE ↓")

    ax.set_xticks(range(len(domain_names)))
    ax.set_yticks(range(len(method_names)))
    ax.set_xticklabels(domain_names, rotation=30, ha="right")
    ax.set_yticklabels(method_names)
    ax.set_title(title)

    # Cell annotations
    for i in range(len(method_names)):
        for j in range(len(domain_names)):
            ax.text(j, i, f"{mae_matrix[i, j]:.3f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if mae_matrix[i, j] > 0.5 * mae_matrix.max() else "black")

    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_relative_improvements(
    method_names:  Sequence[str],
    rel_impr:      Sequence[float],
    baseline_name: str = "AC",
    save_path:     Optional[str | Path] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of relative MAE reduction vs a baseline.
    Positive = better than baseline (green), negative = worse (red).
    """
    ri     = np.array(rel_impr)
    colors = [PALETTE[2] if v >= 0 else PALETTE[0] for v in ri]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(method_names))))
    y_pos   = np.arange(len(method_names))

    ax.barh(y_pos, ri, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(method_names)
    ax.set_xlabel(f"Relative MAE Reduction vs {baseline_name} (%)")
    ax.set_title(f"Improvement over {baseline_name} baseline")
    ax.invert_yaxis()

    _save_or_show(fig, save_path)
    return fig
