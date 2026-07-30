"""
Report generation.

Aggregates experiment results and produces:
* Console-printed summary tables
* CSV files for import into spreadsheets
* LaTeX-formatted tables suitable for papers
* JSON dumps of all raw numbers

The main entry point is ``ExperimentReport``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


class ExperimentReport:
    """
    Aggregates results from one or more ``run_experiment`` calls and
    generates tables and plots.

    Usage::

        report = ExperimentReport(dataset="pacs", backbone="resnet50")
        report.add_result(
            method="DoC",
            target_domain="art_painting",
            predicted_acc=0.71,
            actual_acc=0.68,
        )
        report.save("results/pacs/")
        report.print_summary()
    """

    def __init__(self, dataset: str, backbone: str) -> None:
        self.dataset  = dataset
        self.backbone = backbone
        self._rows: List[Dict] = []

    def add_result(
        self,
        method:        str,
        target_domain: str,
        predicted_acc: float,
        actual_acc:    float,
        score:         Optional[float] = None,
        extra:         Optional[Dict]  = None,
    ) -> None:
        """Record one (method, domain) prediction result."""
        row = {
            "dataset":       self.dataset,
            "backbone":      self.backbone,
            "method":        method,
            "target_domain": target_domain,
            "predicted_acc": predicted_acc,
            "actual_acc":    actual_acc,
            "abs_error":     abs(predicted_acc - actual_acc),
            "score":         score,
        }
        if extra:
            row.update(extra)
        self._rows.append(row)

    # ── Summary table ─────────────────────────────────────────────────────────

    def summary_by_method(self) -> Dict[str, Dict]:
        """
        Compute per-method summary statistics (mean MAE, std MAE, mean
        Pearson r) across target domains.

        Returns
        -------
        dict: method → {"mae_mean", "mae_std", "mae_by_domain"}
        """
        from collections import defaultdict
        by_method: Dict[str, List] = defaultdict(list)
        for row in self._rows:
            by_method[row["method"]].append(row["abs_error"])

        summary = {}
        for method, errs in sorted(by_method.items(), key=lambda x: np.mean(x[1])):
            errs = np.array(errs)
            summary[method] = {
                "mae_mean": float(errs.mean()),
                "mae_std":  float(errs.std()),
                "n":        len(errs),
            }
        return summary

    def print_summary(self) -> None:
        """Print a formatted summary table to stdout."""
        summ = self.summary_by_method()
        col_w = max(len(m) for m in summ) + 2

        header = f"{'Method':<{col_w}}  {'MAE':>8}  {'± Std':>8}  n"
        print(f"\n{'='*len(header)}")
        print(f"Dataset: {self.dataset.upper()}  Backbone: {self.backbone}")
        print('='*len(header))
        print(header)
        print('-'*len(header))
        for method, s in summ.items():
            print(f"{method:<{col_w}}  {s['mae_mean']:8.4f}  "
                  f"{s['mae_std']:8.4f}  {s['n']}")
        print(f"{'='*len(header)}\n")

    def print_per_domain(self) -> None:
        """Print per-domain results for each method."""
        domains  = sorted({r["target_domain"] for r in self._rows})
        methods  = sorted({r["method"]        for r in self._rows})
        col_w    = max(len(m) for m in methods) + 2
        dom_w    = max(len(d) for d in domains) + 2

        print(f"\n{'='*(col_w + dom_w * len(domains) + 10)}")
        print(f"Per-Domain MAE  |  Dataset: {self.dataset.upper()}")
        header = f"{'Method':<{col_w}}" + "".join(f"  {d[:dom_w-2]:<{dom_w}}" for d in domains)
        print(header)
        print('-' * len(header))

        # Build lookup
        lookup: Dict[tuple, float] = {}
        for r in self._rows:
            lookup[(r["method"], r["target_domain"])] = r["abs_error"]

        for method in methods:
            row_str = f"{method:<{col_w}}"
            for d in domains:
                val = lookup.get((method, d), float("nan"))
                row_str += f"  {val:>{dom_w}.4f}"
            print(row_str)
        print()

    # ── LaTeX output ─────────────────────────────────────────────────────────

    def to_latex(self) -> str:
        """
        Return a LaTeX table of per-method mean MAE ± std, formatted
        to match the paper's style.
        """
        summ = self.summary_by_method()
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Accuracy Prediction MAE on " + self.dataset.upper() + r"}",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"Method & MAE (mean) & MAE (std) \\",
            r"\midrule",
        ]
        for method, s in summ.items():
            lines.append(
                f"{method} & {s['mae_mean']:.4f} & {s['mae_std']:.4f} \\\\"
            )
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    # ── Save to disk ──────────────────────────────────────────────────────────

    def save(self, output_dir: str | Path) -> None:
        """
        Save all results to ``output_dir``:
        * ``results.json`` – raw rows
        * ``summary.csv``  – aggregated per-method MAE
        * ``summary.tex``  – LaTeX table
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Raw JSON
        with open(out / "results.json", "w") as f:
            json.dump(self._rows, f, indent=2)

        # Summary CSV
        summ = self.summary_by_method()
        csv_path = out / "summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method", "mae_mean", "mae_std", "n"])
            writer.writeheader()
            for method, s in summ.items():
                writer.writerow({"method": method, **s})

        # LaTeX
        with open(out / "summary.tex", "w") as f:
            f.write(self.to_latex())

        print(f"Results saved → {out.resolve()}")

    @classmethod
    def load(cls, results_json: str | Path) -> "ExperimentReport":
        with open(results_json) as f:
            rows = json.load(f)
        if not rows:
            raise ValueError("Empty results file.")
        report = cls(dataset=rows[0]["dataset"], backbone=rows[0]["backbone"])
        report._rows = rows
        return report
