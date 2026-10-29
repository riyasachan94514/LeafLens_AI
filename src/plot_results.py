"""Publication-quality research plots for plant disease generalization and abstention.

Generates three figures from existing experiment artifacts:
  1. generalization_gap.png: In-domain vs out-of-domain baseline accuracy comparison.
  2. plantdoc_risk_coverage.png: Risk-coverage curve (coverage vs selective accuracy)
     for open and restricted PlantDoc evaluations with threshold labels.
  3. plantdoc_threshold_tradeoff.png: Coverage and selective accuracy as a function
     of confidence threshold for open and restricted PlantDoc evaluations.

Uses matplotlib only (no seaborn). Preserves reported numbers without modification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# --- Aesthetics and Styling Tokens -------------------------------------------
COLOR_PV = "#2a78d6"          # Blue for PlantVillage (in-domain)
COLOR_OPEN = "#eb6834"        # Orange for PlantDoc Open
COLOR_RESTRICTED = "#0ca678"  # Teal for PlantDoc Restricted
COLOR_VAL = "#7950f2"         # Purple for Validation-selected threshold
COLOR_INK = "#1a1a1a"         # Dark neutral for headings/text
COLOR_MUTED = "#5c5c5c"       # Muted text for captions/ticks
COLOR_GRID = "#e6e6e6"        # Subtle grid line
COLOR_SURFACE = "#ffffff"     # White canvas


def apply_publication_style() -> None:
    """Set global matplotlib style tokens for publication-quality rendering."""
    plt.rcParams.update(
        {
            "figure.facecolor": COLOR_SURFACE,
            "axes.facecolor": COLOR_SURFACE,
            "savefig.facecolor": COLOR_SURFACE,
            "axes.edgecolor": "#cccccc",
            "axes.labelcolor": COLOR_INK,
            "axes.titlecolor": COLOR_INK,
            "text.color": COLOR_INK,
            "xtick.color": COLOR_MUTED,
            "ytick.color": COLOR_MUTED,
            "grid.color": COLOR_GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#e0e0e0",
        }
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cross-eval-json",
        type=Path,
        default=Path("results/resnet18_baseline/cross_eval.json"),
        help="Path to baseline cross_eval.json artifact.",
    )
    ap.add_argument(
        "--abstention-json",
        type=Path,
        default=Path("results/resnet18_baseline/abstention/summary.json"),
        help="Path to abstention summary.json artifact.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/resnet18_baseline/figures"),
        help="Directory to save generated plots (default: results/resnet18_baseline/figures).",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution of output figures (minimum 200 DPI, default 300 DPI).",
    )
    return ap.parse_args()


# --- Validation Helpers ------------------------------------------------------
def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact file does not exist: {path.resolve()}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in artifact file {path}: {exc}") from exc
    return data


def validate_cross_eval(data: dict[str, Any], path: Path) -> None:
    if "results" not in data or not isinstance(data["results"], dict):
        raise KeyError(f"Missing or invalid 'results' key in {path}")
    req_splits = ["plantvillage_test_shared", "plantdoc_open", "plantdoc_restricted"]
    for split in req_splits:
        if split not in data["results"]:
            raise KeyError(f"Missing evaluation split '{split}' in {path}['results']")
        if "accuracy" not in data["results"][split]:
            raise KeyError(f"Missing 'accuracy' field for split '{split}' in {path}")


def validate_abstention(data: dict[str, Any], path: Path) -> None:
    if "results" not in data or not isinstance(data["results"], dict):
        raise KeyError(f"Missing or invalid 'results' key in {path}")
    req_splits = ["plantdoc_open", "plantdoc_restricted"]
    for split in req_splits:
        if split not in data["results"]:
            raise KeyError(f"Missing evaluation split '{split}' in {path}['results']")
        sdata = data["results"][split]
        if "fixed_threshold_sweep" not in sdata or not isinstance(sdata["fixed_threshold_sweep"], list):
            raise KeyError(f"Missing or invalid 'fixed_threshold_sweep' in {path}['results']['{split}']")
        if "validation_selected" not in sdata or not isinstance(sdata["validation_selected"], dict):
            raise KeyError(f"Missing or invalid 'validation_selected' in {path}['results']['{split}']")
        for item in sdata["fixed_threshold_sweep"]:
            for f in ["threshold", "coverage", "selective_accuracy"]:
                if f not in item:
                    raise KeyError(f"Missing field '{f}' in {split} fixed_threshold_sweep item in {path}")
        for f in ["threshold", "coverage", "selective_accuracy"]:
            if f not in sdata["validation_selected"]:
                raise KeyError(f"Missing field '{f}' in {split} validation_selected in {path}")


# --- Plot 1: Generalization Gap ----------------------------------------------
def plot_generalization_gap(
    cross_eval: dict[str, Any],
    out_path: Path,
    dpi: int,
) -> dict[str, Any]:
    """Compare PlantVillage shared test accuracy with PlantDoc open and restricted accuracy."""
    res = cross_eval["results"]
    pv_acc = float(res["plantvillage_test_shared"]["accuracy"])
    pd_open_acc = float(res["plantdoc_open"]["accuracy"])
    pd_restr_acc = float(res["plantdoc_restricted"]["accuracy"])

    pv_samples = res["plantvillage_test_shared"].get("n_samples", 5781)
    pd_open_samples = res["plantdoc_open"].get("n_samples", 2569)
    pd_restr_samples = res["plantdoc_restricted"].get("n_samples", 2569)

    labels = [
        f"PlantVillage Test\nShared Classes\n(28 classes, N={pv_samples:,})",
        f"PlantDoc Open\nField Photos\n(39-way, N={pd_open_samples:,})",
        f"PlantDoc Restricted\nMasked to Shared\n(28-way, N={pd_restr_samples:,})",
    ]
    accuracies = [pv_acc, pd_open_acc, pd_restr_acc]
    colors = [COLOR_PV, COLOR_OPEN, COLOR_RESTRICTED]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(labels))
    width = 0.52

    bars = ax.bar(x, accuracies, width=width, color=colors, edgecolor="#ffffff", linewidth=1.5, zorder=3)

    # Add numeric percentage badges on top of bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{acc * 100:.2f}%\n({acc:.4f})",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=COLOR_INK,
        )

    # Generalization gap annotation between in-domain and field deployment
    gap_drop = (pv_acc - pd_open_acc) * 100.0
    rel_drop = ((pv_acc - pd_open_acc) / pv_acc) * 100.0 if pv_acc else 0.0

    ax.annotate(
        f"Generalization Drop:\n-{gap_drop:.1f} pp absolute\n(-{rel_drop:.1f}% relative)",
        xy=(0.28, pv_acc - 0.05),
        xytext=(0.85, 0.65),
        arrowprops=dict(arrowstyle="->", color=COLOR_OPEN, lw=1.5, connectionstyle="arc3,rad=-0.2"),
        fontsize=9.5,
        fontweight="semibold",
        color=COLOR_OPEN,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff5f0", edgecolor=COLOR_OPEN, alpha=0.9),
    )

    # Restricted mitigation annotation
    gain_restr = (pd_restr_acc - pd_open_acc) * 100.0
    ax.annotate(
        f"Logit Masking:\n+{gain_restr:.1f} pp gain\n(eliminates invalid classes)",
        xy=(1.8, pd_restr_acc + 0.08),
        xytext=(1.35, 0.42),
        arrowprops=dict(arrowstyle="->", color=COLOR_RESTRICTED, lw=1.5, connectionstyle="arc3,rad=0.2"),
        fontsize=9,
        fontweight="semibold",
        color=COLOR_RESTRICTED,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#e6fcf5", edgecolor=COLOR_RESTRICTED, alpha=0.9),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Classification Accuracy", fontsize=11, fontweight="semibold")
    ax.set_ylim(0.0, 1.15)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.yaxis.set_major_formatter(lambda y, _: f"{y * 100:.0f}%")

    ax.set_title(
        "Out-of-Domain Generalization Collapse (ResNet-18 Baseline)",
        fontsize=13,
        fontweight="bold",
        pad=16,
    )

    # Clarifying subtitle
    fig.text(
        0.5,
        0.91,
        "Comparison of in-domain laboratory performance vs. unconstrained field photographs",
        ha="center",
        fontsize=9.5,
        color=COLOR_MUTED,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    return {
        "plantvillage_test_shared_accuracy": pv_acc,
        "plantdoc_open_accuracy": pd_open_acc,
        "plantdoc_restricted_accuracy": pd_restr_acc,
        "absolute_drop_pp": gap_drop,
        "relative_drop_pct": rel_drop,
    }


# --- Plot 2: PlantDoc Risk-Coverage Curve -------------------------------------
def plot_plantdoc_risk_coverage(
    abstention: dict[str, Any],
    out_path: Path,
    dpi: int,
) -> dict[str, Any]:
    """Plot coverage on the x-axis and selective accuracy on the y-axis for PlantDoc."""
    res = abstention["results"]
    pd_open = res["plantdoc_open"]
    pd_restr = res["plantdoc_restricted"]

    # Extract Open data
    open_sweep = sorted(pd_open["fixed_threshold_sweep"], key=lambda x: x["coverage"])
    open_cov = [float(x["coverage"]) for x in open_sweep]
    open_acc = [float(x["selective_accuracy"]) for x in open_sweep]
    open_thresh = [float(x["threshold"]) for x in open_sweep]

    open_val = pd_open["validation_selected"]
    open_val_t = float(open_val["threshold"])
    open_val_cov = float(open_val["coverage"])
    open_val_acc = float(open_val["selective_accuracy"])
    open_base_acc = float(open_val["baseline_accuracy"])

    # Extract Restricted data
    restr_sweep = sorted(pd_restr["fixed_threshold_sweep"], key=lambda x: x["coverage"])
    restr_cov = [float(x["coverage"]) for x in restr_sweep]
    restr_acc = [float(x["selective_accuracy"]) for x in restr_sweep]
    restr_thresh = [float(x["threshold"]) for x in restr_sweep]

    restr_val = pd_restr["validation_selected"]
    restr_val_t = float(restr_val["threshold"])
    restr_val_cov = float(restr_val["coverage"])
    restr_val_acc = float(restr_val["selective_accuracy"])
    restr_base_acc = float(restr_val["baseline_accuracy"])

    fig, ax = plt.subplots(figsize=(8.5, 6))

    # Baseline reference lines (τ = 0, full coverage)
    ax.axhline(
        y=restr_base_acc,
        color=COLOR_RESTRICTED,
        linestyle=":",
        linewidth=1.2,
        alpha=0.7,
        label=f"PlantDoc Restricted Baseline ({restr_base_acc * 100:.2f}%)",
    )
    ax.axhline(
        y=open_base_acc,
        color=COLOR_OPEN,
        linestyle=":",
        linewidth=1.2,
        alpha=0.7,
        label=f"PlantDoc Open Baseline ({open_base_acc * 100:.2f}%)",
    )

    # Plot lines with markers
    ax.plot(
        restr_cov,
        restr_acc,
        marker="s",
        markersize=6.5,
        linewidth=2.0,
        color=COLOR_RESTRICTED,
        label="PlantDoc Restricted (28 shared classes)",
        zorder=3,
    )
    ax.plot(
        open_cov,
        open_acc,
        marker="o",
        markersize=6.5,
        linewidth=2.0,
        color=COLOR_OPEN,
        label="PlantDoc Open (39-way classification)",
        zorder=3,
    )

    # Highlight Validation-selected thresholds
    ax.scatter(
        [open_val_cov],
        [open_val_acc],
        color=COLOR_OPEN,
        edgecolor=COLOR_VAL,
        linewidth=2.2,
        s=120,
        marker="D",
        zorder=5,
        label=f"Validation-Selected (Open: τ*={open_val_t:.2f})",
    )
    ax.scatter(
        [restr_val_cov],
        [restr_val_acc],
        color=COLOR_RESTRICTED,
        edgecolor=COLOR_VAL,
        linewidth=2.2,
        s=120,
        marker="D",
        zorder=5,
        label=f"Validation-Selected (Restricted: τ*={restr_val_t:.2f})",
    )

    # Label each point with its threshold
    for c, a, t in zip(open_cov, open_acc, open_thresh):
        ax.annotate(
            f"τ={t:.2f}",
            (c, a),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=8.5,
            color=COLOR_OPEN,
            fontweight="semibold",
        )

    for c, a, t in zip(restr_cov, restr_acc, restr_thresh):
        ax.annotate(
            f"τ={t:.2f}",
            (c, a),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8.5,
            color=COLOR_RESTRICTED,
            fontweight="semibold",
        )

    # Special callouts for validation-selected points
    ax.annotate(
        f"τ*={open_val_t:.2f} (Val)\nAcc={open_val_acc*100:.1f}%\nCov={open_val_cov*100:.1f}%",
        (open_val_cov, open_val_acc),
        textcoords="offset points",
        xytext=(38, 14),
        ha="left",
        fontsize=8.5,
        fontweight="bold",
        color=COLOR_OPEN,
        arrowprops=dict(arrowstyle="->", color=COLOR_OPEN, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor=COLOR_OPEN, alpha=0.9),
    )

    ax.annotate(
        f"τ*={restr_val_t:.2f} (Val)\nAcc={restr_val_acc*100:.1f}%\nCov={restr_val_cov*100:.1f}%",
        (restr_val_cov, restr_val_acc),
        textcoords="offset points",
        xytext=(36, 16),
        ha="left",
        fontsize=8.5,
        fontweight="bold",
        color=COLOR_RESTRICTED,
        arrowprops=dict(arrowstyle="->", color=COLOR_RESTRICTED, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor=COLOR_RESTRICTED, alpha=0.9),
    )

    ax.set_xlabel("Coverage (Fraction of Accepted Samples)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Selective Accuracy Among Accepted Predictions", fontsize=11, fontweight="semibold")
    ax.set_xlim(0.05, 0.75)
    ax.set_ylim(0.10, 0.56)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x * 100:.0f}%")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y * 100:.0f}%")

    ax.set_title(
        "PlantDoc Selective Prediction: Risk-Coverage Trade-Off",
        fontsize=13,
        fontweight="bold",
        pad=16,
    )
    fig.text(
        0.5,
        0.91,
        "Selective accuracy evaluated strictly among accepted samples (excluding abstentions)",
        ha="center",
        fontsize=9.5,
        color=COLOR_MUTED,
    )

    ax.legend(loc="upper right", frameon=True, fontsize=8.5)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    return {
        "open_baseline_accuracy": open_base_acc,
        "open_val_selected_threshold": open_val_t,
        "open_val_selected_coverage": open_val_cov,
        "open_val_selected_selective_accuracy": open_val_acc,
        "open_sweep_thresholds": open_thresh,
        "restricted_baseline_accuracy": restr_base_acc,
        "restricted_val_selected_threshold": restr_val_t,
        "restricted_val_selected_coverage": restr_val_cov,
        "restricted_val_selected_selective_accuracy": restr_val_acc,
        "restricted_sweep_thresholds": restr_thresh,
    }


# --- Plot 3: PlantDoc Threshold Trade-off -------------------------------------
def plot_plantdoc_threshold_tradeoff(
    abstention: dict[str, Any],
    out_path: Path,
    dpi: int,
) -> dict[str, Any]:
    """Plot coverage and selective accuracy against threshold for Open and Restricted PlantDoc."""
    res = abstention["results"]
    pd_open = res["plantdoc_open"]
    pd_restr = res["plantdoc_restricted"]

    # Open
    open_sweep = sorted(pd_open["fixed_threshold_sweep"], key=lambda x: x["threshold"])
    t_open = [float(x["threshold"]) for x in open_sweep]
    cov_open = [float(x["coverage"]) for x in open_sweep]
    acc_open = [float(x["selective_accuracy"]) for x in open_sweep]
    val_t_open = float(pd_open["validation_selected"]["threshold"])
    val_cov_open = float(pd_open["validation_selected"]["coverage"])
    val_acc_open = float(pd_open["validation_selected"]["selective_accuracy"])

    # Restricted
    restr_sweep = sorted(pd_restr["fixed_threshold_sweep"], key=lambda x: x["threshold"])
    t_restr = [float(x["threshold"]) for x in restr_sweep]
    cov_restr = [float(x["coverage"]) for x in restr_sweep]
    acc_restr = [float(x["selective_accuracy"]) for x in restr_sweep]
    val_t_restr = float(pd_restr["validation_selected"]["threshold"])
    val_cov_restr = float(pd_restr["validation_selected"]["coverage"])
    val_acc_restr = float(pd_restr["validation_selected"]["selective_accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)

    # Panel 1: PlantDoc Open
    ax1.plot(
        t_open,
        cov_open,
        marker="s",
        linestyle="--",
        color="#7048e8",
        linewidth=1.8,
        label="Coverage",
        zorder=3,
    )
    ax1.plot(
        t_open,
        acc_open,
        marker="o",
        linestyle="-",
        color=COLOR_OPEN,
        linewidth=2.0,
        label="Selective Accuracy",
        zorder=3,
    )

    # Validation vertical line in Panel 1
    ax1.axvline(
        x=val_t_open,
        color=COLOR_MUTED,
        linestyle=":",
        linewidth=1.4,
        label=f"Val Threshold (τ*={val_t_open:.2f})",
    )
    ax1.scatter([val_t_open], [val_cov_open], color="#7048e8", s=60, zorder=4)
    ax1.scatter([val_t_open], [val_acc_open], color=COLOR_OPEN, s=60, zorder=4)

    ax1.annotate(
        f"τ*={val_t_open:.2f}\nCov: {val_cov_open*100:.1f}%\nAcc: {val_acc_open*100:.1f}%",
        xy=(val_t_open, val_acc_open),
        xytext=(val_t_open - 0.22, 0.40),
        arrowprops=dict(arrowstyle="->", color=COLOR_MUTED, lw=1.2),
        fontsize=8.5,
        fontweight="semibold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.9),
    )

    ax1.set_title("PlantDoc Open (39 Classes)", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Confidence Threshold (τ)", fontsize=10.5, fontweight="semibold")
    ax1.set_ylabel("Metric Value (Fraction / Rate)", fontsize=10.5, fontweight="semibold")
    ax1.set_xlim(0.46, 0.94)
    ax1.set_ylim(0.05, 0.75)
    ax1.yaxis.set_major_formatter(lambda y, _: f"{y * 100:.0f}%")
    ax1.legend(loc="upper left", fontsize=8.5)

    # Panel 2: PlantDoc Restricted
    ax2.plot(
        t_restr,
        cov_restr,
        marker="s",
        linestyle="--",
        color="#7048e8",
        linewidth=1.8,
        label="Coverage",
        zorder=3,
    )
    ax2.plot(
        t_restr,
        acc_restr,
        marker="o",
        linestyle="-",
        color=COLOR_RESTRICTED,
        linewidth=2.0,
        label="Selective Accuracy",
        zorder=3,
    )

    # Validation vertical line in Panel 2
    ax2.axvline(
        x=val_t_restr,
        color=COLOR_MUTED,
        linestyle=":",
        linewidth=1.4,
        label=f"Val Threshold (τ*={val_t_restr:.2f})",
    )
    ax2.scatter([val_t_restr], [val_cov_restr], color="#7048e8", s=60, zorder=4)
    ax2.scatter([val_t_restr], [val_acc_restr], color=COLOR_RESTRICTED, s=60, zorder=4)

    ax2.annotate(
        f"τ*={val_t_restr:.2f}\nCov: {val_cov_restr*100:.1f}%\nAcc: {val_acc_restr*100:.1f}%",
        xy=(val_t_restr, val_acc_restr),
        xytext=(val_t_restr - 0.22, 0.20),
        arrowprops=dict(arrowstyle="->", color=COLOR_MUTED, lw=1.2),
        fontsize=8.5,
        fontweight="semibold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.9),
    )

    ax2.set_title("PlantDoc Restricted (28 Shared Classes)", fontsize=11.5, fontweight="bold")
    ax2.set_xlabel("Confidence Threshold (τ)", fontsize=10.5, fontweight="semibold")
    ax2.set_xlim(0.46, 0.94)
    ax2.legend(loc="upper left", fontsize=8.5)

    fig.suptitle(
        "Coverage vs. Selective Accuracy Across Confidence Thresholds",
        fontsize=13,
        fontweight="bold",
        y=0.97,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    return {
        "open_thresholds": t_open,
        "open_coverage": cov_open,
        "open_selective_accuracy": acc_open,
        "open_validation_threshold": val_t_open,
        "restricted_thresholds": t_restr,
        "restricted_coverage": cov_restr,
        "restricted_selective_accuracy": acc_restr,
        "restricted_validation_threshold": val_t_restr,
    }


# --- Main Orchestration ------------------------------------------------------
def main() -> None:
    args = parse_args()
    apply_publication_style()

    # Ensure output directory exists
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Plant Disease Generalization: Research Plot Generation")
    print("=" * 80)
    print(f"Reading cross-evaluation artifact: {args.cross_eval_json}")
    print(f"Reading abstention artifact:       {args.abstention_json}")
    print(f"Output directory:                  {args.out_dir}")
    print(f"Resolution:                        {args.dpi} DPI")
    print("-" * 80)

    # Load and validate JSON artifacts
    cross_eval = load_json(args.cross_eval_json)
    validate_cross_eval(cross_eval, args.cross_eval_json)

    abstention = load_json(args.abstention_json)
    validate_abstention(abstention, args.abstention_json)

    # 1. Generalization Gap Plot
    p1_path = args.out_dir / "generalization_gap.png"
    p1_metrics = plot_generalization_gap(cross_eval, p1_path, dpi=args.dpi)
    print(f"[Figure 1 Saved] {p1_path}")
    print("  Source: cross_eval.json['results']")
    print(f"  - PlantVillage shared accuracy: {p1_metrics['plantvillage_test_shared_accuracy']:.4f} ({p1_metrics['plantvillage_test_shared_accuracy']*100:.2f}%)")
    print(f"  - PlantDoc open accuracy:        {p1_metrics['plantdoc_open_accuracy']:.4f} ({p1_metrics['plantdoc_open_accuracy']*100:.2f}%)")
    print(f"  - PlantDoc restricted accuracy:  {p1_metrics['plantdoc_restricted_accuracy']:.4f} ({p1_metrics['plantdoc_restricted_accuracy']*100:.2f}%)")
    print(f"  - Absolute drop:                 {p1_metrics['absolute_drop_pp']:.2f} pp (Relative: {p1_metrics['relative_drop_pct']:.2f}%)")
    print()

    # 2. Risk-Coverage Plot
    p2_path = args.out_dir / "plantdoc_risk_coverage.png"
    p2_metrics = plot_plantdoc_risk_coverage(abstention, p2_path, dpi=args.dpi)
    print(f"[Figure 2 Saved] {p2_path}")
    print("  Source: abstention/summary.json['results']['plantdoc_open' & 'plantdoc_restricted']")
    print(f"  - Open baseline accuracy:       {p2_metrics['open_baseline_accuracy']:.4f}")
    print(f"  - Open val threshold (tau*):      {p2_metrics['open_val_selected_threshold']:.4f} -> Cov: {p2_metrics['open_val_selected_coverage']*100:.2f}%, Sel-Acc: {p2_metrics['open_val_selected_selective_accuracy']*100:.2f}%")
    print(f"  - Restricted baseline accuracy: {p2_metrics['restricted_baseline_accuracy']:.4f}")
    print(f"  - Restricted val threshold (tau*):{p2_metrics['restricted_val_selected_threshold']:.4f} -> Cov: {p2_metrics['restricted_val_selected_coverage']*100:.2f}%, Sel-Acc: {p2_metrics['restricted_val_selected_selective_accuracy']*100:.2f}%")
    print()

    # 3. Threshold Trade-off Plot
    p3_path = args.out_dir / "plantdoc_threshold_tradeoff.png"
    p3_metrics = plot_plantdoc_threshold_tradeoff(abstention, p3_path, dpi=args.dpi)
    print(f"[Figure 3 Saved] {p3_path}")
    print("  Source: abstention/summary.json['results']['plantdoc_open' & 'plantdoc_restricted']")
    print(f"  - Open threshold range:         {min(p3_metrics['open_thresholds'])} to {max(p3_metrics['open_thresholds'])}")
    print(f"  - Restricted threshold range:   {min(p3_metrics['restricted_thresholds'])} to {max(p3_metrics['restricted_thresholds'])}")
    print("=" * 80)
    print("Plot generation completed successfully.")


if __name__ == "__main__":
    main()
