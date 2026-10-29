# Confidence-Aware Selective Prediction (Abstention)

## Overview

Selective prediction (classification with a reject/abstention option) allows a classifier to output a prediction only when its confidence meets or exceeds a predefined threshold $\tau$, and abstain otherwise:

$$\hat{y}(x) = \begin{cases} \arg\max_k f_k(x) & \text{if } \max_k \text{softmax}(f(x))_k \ge \tau \\ \text{abstain} & \text{otherwise} \end{cases}$$

This technique is evaluated here on the frozen `resnet18_baseline` checkpoint (`results/resnet18_baseline/best.pt`) trained on PlantVillage (39 classes) and tested under domain shift on PlantDoc (field photographs).

---

## Threshold Selection Methodology

**Constraint:** The confidence threshold $\tau^*$ is chosen strictly using **PlantVillage validation data only**. Out-of-domain PlantDoc labels and images were never consulted, optimized against, or observed during threshold selection.

### Threshold Selection Rule: Bounded Risk (Target Accuracy)
- **Rule:** Find the lowest confidence threshold $\tau \in [0, 1]$ such that the selective accuracy on the PlantVillage validation set satisfies:
  $$\text{SelectiveAccuracy}_{\text{val}}(\tau) \ge 0.9980$$
- **Rationale:** The baseline validation accuracy is 0.9952 (40 errors across 8,317 images). Setting an in-domain target of 0.9980 cuts the validation error rate by more than half while retaining high coverage. By picking the minimal threshold satisfying this bound, coverage is maximized subject to the target risk constraint.
- **Result on PlantVillage Validation:**
  - Selected threshold: $\tau^* = 0.7969$
  - Validation samples: 8,317
  - Baseline validation accuracy: 99.52%
  - Validation coverage: 98.68% (8,207 accepted, 110 abstained)
  - Validation selective accuracy: 99.82%
  - Validation abstention rate: 1.32%

---

## Empirical Results

Evaluated on the frozen model across:
1. `plantvillage_test`: Full 39-class held-out test split (8,319 images)
2. `plantvillage_test_shared`: 28 shared classes held-out test split (5,781 images)
3. `plantdoc_open`: Field images evaluated over the full 39-class output head (2,569 images)
4. `plantdoc_restricted`: Field images with logits masked to the 28 shared classes (2,569 images)

### 1. Validation-Selected Threshold ($\tau^* = 0.7969$)

| Dataset | Total Samples | Baseline Acc | Coverage | Accepted Samples | Abstained Samples | Abstention Rate | Selective Acc | Macro-F1 |
|---|---|---|---|---|---|---|---|---|
| `plantvillage_test` | 8,319 | 99.62% | 98.94% | 8,231 | 88 | 1.06% | **99.91%** | 0.9985 |
| `plantvillage_test_shared` | 5,781 | 99.53% | 98.77% | 5,710 | 71 | 1.23% | **99.89%** | 0.9981 |
| `plantdoc_open` | 2,569 | 13.78% | 35.31% | 907 | 1,662 | 64.69% | **16.21%** | 0.1575 |
| `plantdoc_restricted` | 2,569 | 22.89% | 18.92% | 486 | 2,083 | 81.08% | **43.42%** | 0.3166 |

### 2. Predefined Fixed Threshold Sweep

#### PlantVillage Test (39 classes, N = 8,319, Baseline Acc = 99.62%)
| Threshold | Coverage | Abstention Rate | Accepted Samples | Selective Acc | Macro-F1 |
|---|---|---|---|---|---|
| 0.50 | 99.83% | 0.17% | 8,305 | 99.70% | 0.9953 |
| 0.60 | 99.52% | 0.48% | 8,279 | 99.84% | 0.9974 |
| 0.70 | 99.36% | 0.64% | 8,266 | 99.88% | 0.9980 |
| 0.80 | 98.92% | 1.08% | 8,229 | 99.91% | 0.9985 |
| 0.90 | 96.75% | 3.25% | 8,049 | 99.99% | 0.9997 |
| **0.7969 (Val)** | **98.94%** | **1.06%** | **8,231** | **99.91%** | **0.9985** |

#### PlantVillage Test Shared (28 classes, N = 5,781, Baseline Acc = 99.53%)
| Threshold | Coverage | Abstention Rate | Accepted Samples | Selective Acc | Macro-F1 |
|---|---|---|---|---|---|
| 0.50 | 99.78% | 0.22% | 5,768 | 99.65% | 0.9952 |
| 0.60 | 99.43% | 0.57% | 5,748 | 99.81% | 0.9969 |
| 0.70 | 99.22% | 0.78% | 5,736 | 99.84% | 0.9974 |
| 0.80 | 98.75% | 1.25% | 5,709 | 99.89% | 0.9981 |
| 0.90 | 96.23% | 3.77% | 5,563 | 99.98% | 0.9995 |
| **0.7969 (Val)** | **98.77%** | **1.23%** | **5,710** | **99.89%** | **0.9981** |

#### PlantDoc Open (39 classes, N = 2,569, Baseline Acc = 13.78%)
| Threshold | Coverage | Abstention Rate | Accepted Samples | Selective Acc | Macro-F1 |
|---|---|---|---|---|---|
| 0.50 | 68.63% | 31.37% | 1,763 | 14.18% | 0.1453 |
| 0.60 | 57.69% | 42.31% | 1,482 | 14.51% | 0.1422 |
| 0.70 | 46.75% | 53.25% | 1,201 | 15.24% | 0.1493 |
| 0.80 | 34.88% | 65.12% | 896 | 16.18% | 0.1557 |
| 0.90 | 19.42% | 80.58% | 499 | 16.83% | 0.1398 |
| **0.7969 (Val)** | **35.31%** | **64.69%** | **907** | **16.21%** | **0.1575** |

#### PlantDoc Restricted (28 classes, N = 2,569, Baseline Acc = 22.89%)
| Threshold | Coverage | Abstention Rate | Accepted Samples | Selective Acc | Macro-F1 |
|---|---|---|---|---|---|
| 0.50 | 44.57% | 55.43% | 1,145 | 30.92% | 0.2318 |
| 0.60 | 34.64% | 65.36% | 890 | 33.48% | 0.2541 |
| 0.70 | 25.89% | 74.11% | 665 | 38.65% | 0.2907 |
| 0.80 | 18.72% | 81.28% | 481 | 43.66% | 0.3209 |
| 0.90 | 10.82% | 89.18% | 278 | 50.72% | 0.3292 |
| **0.7969 (Val)** | **18.92%** | **81.08%** | **486** | **43.42%** | **0.3166** |

---

## Methodological Limitations and Discussion

1. **Abstention Does Not Improve Classifier Accuracy:**
   Abstention is a decision-rule filter, not a training mechanism. The underlying neural network parameters and feature representations are untouched. Any increase in selective accuracy occurs solely because low-confidence predictions are removed from the evaluation pool, reducing coverage.

2. **Severe Out-of-Domain Overconfidence:**
   On in-domain data (PlantVillage), softmax confidence correlates well with correctness: misclassified samples have much lower mean confidence (0.698 vs 0.957). However, on out-of-domain field data (PlantDoc), the model exhibits severe overconfidence on incorrect predictions (e.g. predicting `Background_without_leaves` with high probability). As a result, in `plantdoc_open`, rejecting 64.7% of the dataset only improves accuracy from 13.78% to 16.21%.

3. **Restricted Domain Mitigation:**
   When impossibly non-existent training classes are masked out (`plantdoc_restricted`), softmax probability redistribution allows the threshold to more effectively discard errors, reaching 43.42% accuracy at 18.92% coverage. Nevertheless, more than 81% of field samples must be abandoned to achieve this level of reliability.

4. **Empty Subset Handling:**
   If a high threshold rejects all samples ($N_{\text{accepted}} = 0$), selective accuracy, balanced accuracy, and macro-F1 evaluate safely to `None` (JSON `null`) with coverage $0.0$ and abstention rate $1.0$, preventing division-by-zero errors.

---

## Artifacts and Output Files

All experiment outputs are preserved under `results/resnet18_baseline/abstention/`:
- `summary.json`: Machine-readable metadata, validation threshold selection details, and evaluation metrics across all sweeps.
- `plantvillage_test_predictions.json`: Per-image predictions for full PlantVillage test set (8,319 items).
- `plantvillage_test_shared_predictions.json`: Per-image predictions for shared-class PlantVillage test set (5,781 items).
- `plantdoc_open_predictions.json`: Per-image predictions for open PlantDoc field set (2,569 items).
- `plantdoc_restricted_predictions.json`: Per-image predictions for restricted PlantDoc field set (2,569 items).

Each per-image prediction includes:
`sample_id`, `image_path`, `true_label_id`, `true_label_name`, `predicted_label_id`, `predicted_label_name`, `confidence`, `is_correct`, `abstained_validation_selected`, and `abstained_by_fixed_threshold`.
