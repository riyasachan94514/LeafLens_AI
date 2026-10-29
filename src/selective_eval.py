"""Confidence-aware selective prediction (abstention) evaluation.

Evaluates a frozen PlantVillage-trained classifier under Maximum Softmax
Probability (MSP) confidence thresholding. The decision rule accepts a
prediction if max_i softmax(logits)_i >= threshold and abstains otherwise.

The threshold is selected using PlantVillage validation data ONLY, and never
uses PlantDoc out-of-domain labels or samples.

Evaluates:
  - plantvillage_test: full 39-class PlantVillage held-out test split
  - plantvillage_test_shared: 28 shared-class PlantVillage held-out test split
  - plantdoc_open: full 39-class PlantDoc field photos
  - plantdoc_restricted: PlantDoc field photos with logits masked to the 28 shared classes
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
import torch

from data import build_crosseval_loader, build_loaders, read_vocabulary
from engine import get_device, predict
from labels import (
    PLANTDOC_TO_PLANTVILLAGE,
    assert_mapping_targets_exist,
    crop_of,
    is_healthy,
)
from metrics import _softmax
from models import build_model


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="resnet18", help="Architecture name (default: resnet18).")
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=Path("results/resnet18_baseline"),
        help="Directory holding best.pt checkpoint.",
    )
    ap.add_argument("--data-root", type=Path, default=Path("data"), help="Root data directory.")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <run-dir>/abstention).",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument(
        "--selection-rule",
        choices=["target_accuracy", "target_coverage", "youden_j", "max_f1"],
        default="target_accuracy",
        help="Rule to select confidence threshold on PlantVillage validation split.",
    )
    ap.add_argument(
        "--target-accuracy",
        type=float,
        default=0.998,
        help="Target selective accuracy on val data for target_accuracy rule (default: 0.998).",
    )
    ap.add_argument(
        "--target-coverage",
        type=float,
        default=0.95,
        help="Target coverage on val data for target_coverage rule (default: 0.95).",
    )
    ap.add_argument(
        "--sweep-thresholds",
        nargs="+",
        type=float,
        default=[0.50, 0.60, 0.70, 0.80, 0.90],
        help="Predefined threshold sweep list.",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return ap.parse_args()


def _restrict_logits(logits: np.ndarray, allowed: list[int]) -> np.ndarray:
    """Mask every class outside `allowed` to -inf so it can never be predicted."""
    masked = np.full_like(logits, -np.inf)
    masked[:, allowed] = logits[:, allowed]
    return masked


def compute_msp_confidence(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute maximum softmax probability (MSP) confidence and predicted class."""
    probs = _softmax(logits.astype(np.float64))
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    return confidence, predictions


def select_validation_threshold(
    val_logits: np.ndarray,
    val_targets: np.ndarray,
    rule: str = "target_accuracy",
    target_accuracy: float = 0.998,
    target_coverage: float = 0.95,
) -> dict[str, Any]:
    """Select the confidence threshold using PlantVillage validation data ONLY.

    Rules:
      - target_accuracy: Smallest threshold achieving validation selective accuracy >= target_accuracy.
                         If multiple thresholds achieve the target, picks the lowest to maximize coverage.
      - target_coverage: (1 - target_coverage) quantile of validation confidence.
      - youden_j: Maximizes Youden's J statistic (Sensitivity + Specificity - 1) for error abstention.
      - max_f1: Maximizes F1 score for detecting incorrect predictions.
    """
    confs, preds = compute_msp_confidence(val_logits)
    correct = preds == val_targets
    errors = ~correct
    n_samples = len(val_targets)
    baseline_acc = float(correct.mean())

    selected_threshold: float = 0.50
    rule_description = ""

    if rule == "target_accuracy":
        # Candidate thresholds from 0.0 to 0.999 with 10,000 fine steps
        candidates = np.linspace(0.0, 0.999, 10000)
        found = False
        for t in candidates:
            mask = confs >= t
            if mask.any() and correct[mask].mean() >= target_accuracy:
                selected_threshold = float(t)
                found = True
                break
        if not found:
            # Fall back to threshold that gives highest accuracy
            best_acc = -1.0
            for t in candidates:
                mask = confs >= t
                if mask.any():
                    acc = correct[mask].mean()
                    if acc > best_acc:
                        best_acc = acc
                        selected_threshold = float(t)
        rule_description = (
            f"Bounded risk: lowest threshold yielding validation selective accuracy >= "
            f"{target_accuracy:.4f} (baseline val accuracy is {baseline_acc:.4f})"
        )

    elif rule == "target_coverage":
        pct = (1.0 - target_coverage) * 100.0
        selected_threshold = float(np.percentile(confs, pct))
        rule_description = (
            f"Target coverage: {target_coverage:.2%} percentile cut on validation confidence"
        )

    elif rule == "youden_j":
        best_j = -1.0
        candidates = np.linspace(0.0, 0.999, 10000)
        for t in candidates:
            abstain = confs < t
            tpr = (abstain & errors).sum() / errors.sum() if errors.any() else 0.0
            fpr = (abstain & ~errors).sum() / (~errors).sum() if (~errors).any() else 0.0
            j = tpr - fpr
            if j > best_j:
                best_j = j
                selected_threshold = float(t)
        rule_description = f"Optimal Youden's J ({best_j:.4f}) for abstaining on validation errors"

    elif rule == "max_f1":
        best_f1 = -1.0
        candidates = np.linspace(0.0, 0.999, 10000)
        for t in candidates:
            abstain = confs < t
            tp = (abstain & errors).sum()
            fp = (abstain & ~errors).sum()
            fn = (~abstain & errors).sum()
            denom = 2 * tp + fp + fn
            f1 = (2 * tp / denom) if denom > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                selected_threshold = float(t)
        rule_description = (
            f"Maximum error-detection F1 score ({best_f1:.4f}) on validation set"
        )

    # Calculate validation stats at selected threshold
    val_accepted = confs >= selected_threshold
    val_acc_count = int(val_accepted.sum())
    val_coverage = float(val_acc_count / n_samples)
    val_sel_acc = float(correct[val_accepted].mean()) if val_acc_count > 0 else None

    return {
        "rule": rule,
        "rule_description": rule_description,
        "target_accuracy": target_accuracy if rule == "target_accuracy" else None,
        "target_coverage": target_coverage if rule == "target_coverage" else None,
        "selected_threshold": selected_threshold,
        "val_samples": n_samples,
        "val_baseline_accuracy": baseline_acc,
        "val_coverage": val_coverage,
        "val_selective_accuracy": val_sel_acc,
        "val_abstention_rate": float(1.0 - val_coverage),
        "val_accepted_count": val_acc_count,
        "val_abstained_count": n_samples - val_acc_count,
    }


def compute_selective_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    class_names: list[str],
    threshold: float,
    eligible_label_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Compute selective classification metrics for a single confidence threshold.

    Handles empty accepted subsets safely without division by zero or errors.
    """
    confs, preds = compute_msp_confidence(logits)
    sample_count = len(targets)
    correct = preds == targets
    baseline_acc = float(correct.mean()) if sample_count > 0 else 0.0

    accepted_mask = confs >= threshold
    accepted_samples = int(accepted_mask.sum())
    abstained_samples = sample_count - accepted_samples
    coverage = float(accepted_samples / sample_count) if sample_count > 0 else 0.0
    abstention_rate = float(1.0 - coverage)

    if eligible_label_ids is None:
        eligible_label_ids = list(range(len(class_names)))

    if accepted_samples == 0:
        return {
            "threshold": float(threshold),
            "sample_count": sample_count,
            "baseline_accuracy": baseline_acc,
            "coverage": coverage,
            "selective_accuracy": None,
            "abstention_rate": abstention_rate,
            "accepted_samples": 0,
            "abstained_samples": sample_count,
            "macro_f1": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1_eligible": None,
            "balanced_accuracy": None,
            "crop_accuracy": None,
            "healthy_vs_diseased_accuracy": None,
            "n_classes_active": 0,
            "n_classes_eligible": len(eligible_label_ids),
        }

    acc_targets = targets[accepted_mask]
    acc_preds = preds[accepted_mask]
    selective_acc = float(correct[accepted_mask].mean())

    # Active classes in the accepted subset: union of ground truth and predictions.
    # Includes classes predicted with FP > 0 even if absent from ground truth.
    active_label_ids = sorted(set(acc_targets.tolist()) | set(acc_preds.tolist()))
    macro_acc = precision_recall_fscore_support(
        acc_targets, acc_preds, labels=active_label_ids, average="macro", zero_division=0
    )

    # Eligible catalog classes for this evaluation regime:
    # 28 shared classes for restricted evaluation, 39 classes for open evaluation.
    macro_eligible = precision_recall_fscore_support(
        acc_targets, acc_preds, labels=eligible_label_ids, average="macro", zero_division=0
    )

    balanced_acc = (
        float(balanced_accuracy_score(acc_targets, acc_preds))
        if len(set(acc_targets.tolist())) > 1
        else selective_acc
    )

    # Agronomic relaxed metrics on accepted predictions
    pred_crops = [crop_of(class_names[p]) for p in acc_preds]
    true_crops = [crop_of(class_names[t]) for t in acc_targets]
    crop_acc = float(np.mean([p == t for p, t in zip(pred_crops, true_crops)]))

    pred_healthy = [is_healthy(class_names[p]) for p in acc_preds]
    true_healthy = [is_healthy(class_names[t]) for t in acc_targets]
    healthy_acc = float(np.mean([p == t for p, t in zip(pred_healthy, true_healthy)]))

    return {
        "threshold": float(threshold),
        "sample_count": sample_count,
        "baseline_accuracy": baseline_acc,
        "coverage": coverage,
        "selective_accuracy": selective_acc,
        "abstention_rate": abstention_rate,
        "accepted_samples": accepted_samples,
        "abstained_samples": abstained_samples,
        "macro_f1": float(macro_acc[2]),
        "macro_precision": float(macro_acc[0]),
        "macro_recall": float(macro_acc[1]),
        "macro_f1_eligible": float(macro_eligible[2]),
        "balanced_accuracy": balanced_acc,
        "crop_accuracy": crop_acc,
        "healthy_vs_diseased_accuracy": healthy_acc,
        "n_classes_active": len(active_label_ids),
        "n_classes_eligible": len(eligible_label_ids),
    }


def format_selective_table(eval_name: str, val_selected: dict, sweep: list[dict]) -> str:
    """Format human-readable comparison table."""
    lines = [
        f"==========================================================================================",
        f"Evaluation: {eval_name} (N = {val_selected['sample_count']}, Baseline Acc: {val_selected['baseline_accuracy']:.4f})",
        f"------------------------------------------------------------------------------------------",
        f" Threshold Type         Threshold  Coverage  Abstain%  Accepted  Sel-Acc   Macro-F1  Elig-F1",
        f"------------------------------------------------------------------------------------------",
    ]
    # Fixed thresholds
    for r in sweep:
        sel_acc_str = f"{r['selective_accuracy']:.4f}" if r["selective_accuracy"] is not None else "  N/A "
        macro_str = f"{r['macro_f1']:.4f}" if r["macro_f1"] is not None else "  N/A "
        elig_str = f"{r['macro_f1_eligible']:.4f}" if r.get("macro_f1_eligible") is not None else "  N/A "
        lines.append(
            f" Fixed Sweep             {r['threshold']:7.2f}  "
            f"{r['coverage']:8.4f}  {r['abstention_rate']:8.4f}  "
            f"{r['accepted_samples']:8d}  {sel_acc_str}   {macro_str}   {elig_str}"
        )
    # Validation selected
    sel_acc_str = (
        f"{val_selected['selective_accuracy']:.4f}"
        if val_selected["selective_accuracy"] is not None
        else "  N/A "
    )
    macro_str = (
        f"{val_selected['macro_f1']:.4f}"
        if val_selected["macro_f1"] is not None
        else "  N/A "
    )
    elig_str = (
        f"{val_selected['macro_f1_eligible']:.4f}"
        if val_selected.get("macro_f1_eligible") is not None
        else "  N/A "
    )
    lines.append(
        f" [Validation Selected]   {val_selected['threshold']:7.4f}  "
        f"{val_selected['coverage']:8.4f}  {val_selected['abstention_rate']:8.4f}  "
        f"{val_selected['accepted_samples']:8d}  {sel_acc_str}   {macro_str}   {elig_str}"
    )
    lines.append(
        f"=========================================================================================="
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Device: {device}")

    # Output directory
    out_dir = args.out_dir or (args.run_dir / "abstention")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing abstention evaluation outputs to: {out_dir}")

    # Vocabulary & Shared classes
    vocabulary = read_vocabulary(args.data_root / "plantvillage")
    assert_mapping_targets_exist(vocabulary)
    shared_names = sorted(set(PLANTDOC_TO_PLANTVILLAGE.values()))
    shared_ids = sorted(vocabulary.index(n) for n in shared_names)
    print(f"Total vocabulary: {len(vocabulary)} classes, Shared: {len(shared_ids)} classes")

    # Load frozen model checkpoint
    ckpt_path = args.run_dir / "best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(args.model, len(vocabulary), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded frozen model checkpoint from {ckpt_path} (epoch {checkpoint['epoch']})")

    # -------------------------------------------------------------------------
    # STEP 1: Select threshold using PlantVillage validation data ONLY
    # -------------------------------------------------------------------------
    print("\n[Step 1] Running inference on PlantVillage validation split...")
    val_loaders, _ = build_loaders(
        args.data_root / "plantvillage",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        splits=("val",),
    )
    val_logits, val_targets = predict(model, val_loaders["val"], device)

    threshold_info = select_validation_threshold(
        val_logits,
        val_targets,
        rule=args.selection_rule,
        target_accuracy=args.target_accuracy,
        target_coverage=args.target_coverage,
    )
    tau_star = threshold_info["selected_threshold"]
    print(f"Threshold Selection Rule: {threshold_info['rule_description']}")
    print(f"Selected Threshold tau* = {tau_star:.4f}")
    print(
        f"Validation Set (N = {threshold_info['val_samples']}): "
        f"Baseline Acc = {threshold_info['val_baseline_accuracy']:.4f}, "
        f"Coverage = {threshold_info['val_coverage']:.4f}, "
        f"Selective Acc = {threshold_info['val_selective_accuracy']:.4f}, "
        f"Abstention Rate = {threshold_info['val_abstention_rate']:.4f}"
    )

    # -------------------------------------------------------------------------
    # STEP 2: Run inference on PlantVillage held-out test split
    # -------------------------------------------------------------------------
    print("\n[Step 2] Running inference on PlantVillage test split...")
    test_loaders, _ = build_loaders(
        args.data_root / "plantvillage",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        splits=("test",),
    )
    pv_test_logits, pv_test_targets = predict(model, test_loaders["test"], device)
    pv_test_paths = [s[0] for s in test_loaders["test"].dataset.samples]

    # Subsets: full 39 classes vs 28 shared classes
    pv_shared_mask = np.isin(pv_test_targets, shared_ids)
    pv_shared_logits = pv_test_logits[pv_shared_mask]
    pv_shared_targets = pv_test_targets[pv_shared_mask]
    pv_shared_paths = [p for p, m in zip(pv_test_paths, pv_shared_mask) if m]

    # -------------------------------------------------------------------------
    # STEP 3: Run inference on PlantDoc field photos
    # -------------------------------------------------------------------------
    print("\n[Step 3] Running inference on PlantDoc field split...")
    pd_loader = build_crosseval_loader(
        args.data_root / "plantdoc",
        vocabulary,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    pd_logits, pd_targets = predict(model, pd_loader, device)
    pd_paths = [s[0] for s in pd_loader.dataset.samples]

    pd_restricted_logits = _restrict_logits(pd_logits, shared_ids)

    # -------------------------------------------------------------------------
    # STEP 4: Evaluate all datasets across sweep and validation threshold
    # -------------------------------------------------------------------------
    datasets: dict[str, dict[str, Any]] = {
        "plantvillage_test": {
            "logits": pv_test_logits,
            "targets": pv_test_targets,
            "paths": pv_test_paths,
            "eligible_label_ids": list(range(len(vocabulary))),
            "description": "PlantVillage held-out test split (all 39 classes)",
        },
        "plantvillage_test_shared": {
            "logits": pv_shared_logits,
            "targets": pv_shared_targets,
            "paths": pv_shared_paths,
            "eligible_label_ids": shared_ids,
            "description": "PlantVillage held-out test split restricted to 28 shared classes",
        },
        "plantdoc_open": {
            "logits": pd_logits,
            "targets": pd_targets,
            "paths": pd_paths,
            "eligible_label_ids": list(range(len(vocabulary))),
            "description": "PlantDoc field photos evaluated over open 39-class head",
        },
        "plantdoc_restricted": {
            "logits": pd_restricted_logits,
            "targets": pd_targets,
            "paths": pd_paths,
            "eligible_label_ids": shared_ids,
            "description": "PlantDoc field photos with logits restricted to 28 shared classes",
        },
    }

    results: dict[str, Any] = {}
    sweep_thresholds = sorted(set(args.sweep_thresholds))

    print("\n[Step 4] Evaluating selective prediction...")
    for eval_name, dinfo in datasets.items():
        lgts = dinfo["logits"]
        tgts = dinfo["targets"]
        eligible_ids = dinfo["eligible_label_ids"]

        # 1. Sweep fixed thresholds
        sweep_results: list[dict[str, Any]] = []
        for t in sweep_thresholds:
            metrics = compute_selective_metrics(
                lgts, tgts, vocabulary, threshold=t, eligible_label_ids=eligible_ids
            )
            sweep_results.append(metrics)

        # 2. Validation-selected threshold
        val_selected_metrics = compute_selective_metrics(
            lgts, tgts, vocabulary, threshold=tau_star, eligible_label_ids=eligible_ids
        )

        results[eval_name] = {
            "description": dinfo["description"],
            "sample_count": len(tgts),
            "baseline_accuracy": val_selected_metrics["baseline_accuracy"],
            "validation_selected": val_selected_metrics,
            "fixed_threshold_sweep": sweep_results,
        }

        # Print table to console
        print()
        print(format_selective_table(eval_name, val_selected_metrics, sweep_results))

    # -------------------------------------------------------------------------
    # STEP 5: Export per-image predictions with true, predicted, conf, abstain
    # -------------------------------------------------------------------------
    print("\n[Step 5] Saving per-image predictions...")
    for eval_name, dinfo in datasets.items():
        lgts = dinfo["logits"]
        tgts = dinfo["targets"]
        paths = dinfo["paths"]

        confs, preds = compute_msp_confidence(lgts)
        corrects = preds == tgts

        records: list[dict[str, Any]] = []
        for idx in range(len(tgts)):
            t_id = int(tgts[idx])
            p_id = int(preds[idx])
            conf_val = float(confs[idx])
            abstained_val = bool(conf_val < tau_star)

            abstained_sweep = {
                f"{t:.2f}": bool(conf_val < t) for t in sweep_thresholds
            }

            records.append({
                "sample_id": idx,
                "image_path": str(paths[idx]),
                "true_label_id": t_id,
                "true_label_name": vocabulary[t_id],
                "predicted_label_id": p_id,
                "predicted_label_name": vocabulary[p_id],
                "confidence": conf_val,
                "is_correct": bool(corrects[idx]),
                "abstained_validation_selected": abstained_val,
                "accepted_validation_selected": not abstained_val,
                "abstained_by_fixed_threshold": abstained_sweep,
            })

        pred_file = out_dir / f"{eval_name}_predictions.json"
        pred_file.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"Wrote {len(records)} predictions to {pred_file}")

    # -------------------------------------------------------------------------
    # STEP 6: Save machine-readable summary JSON
    # -------------------------------------------------------------------------
    summary_payload = {
        "metadata": {
            "created_at": datetime.datetime.now().isoformat(),
            "model": args.model,
            "checkpoint": str(ckpt_path),
            "checkpoint_epoch": checkpoint["epoch"],
            "seed": args.seed,
            "threshold_selection": threshold_info,
            "sweep_thresholds": sweep_thresholds,
            "shared_classes": shared_names,
            "total_classes": len(vocabulary),
        },
        "results": results,
    }

    summary_file = out_dir / "summary.json"
    summary_file.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"\nWrote full selective evaluation summary to {summary_file}")


if __name__ == "__main__":
    main()
