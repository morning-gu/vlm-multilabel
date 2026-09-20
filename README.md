# VLM Multi-Label: Label-Token Supervision for Multi-Label Classification

Training framework for multi-label classification with autoregressive VLMs.
The main method is **CE+BCE**:

```text
L_total = L_CE(full sequence) + w_bce * L_BCE(label-token logits)
```

The set-level MP correction is an optional diagnostic/regularizer; CE+BCE is
the default and best-supported objective.

## Primary-label protocol

All converted training/validation records use the **first positive label in
the dataset-config order** as the designated primary label. Training/validation
converters skip no-positive records because they cannot supply a first-positive
CE target. Training also validates every train/val record before starting.

For ChestX-ray14, the converter preserves the **complete official test split**,
including no-positive (`No Finding`) images, whose 14-dim `label_vector` is
all zero. This is required for standard multi-label mAP, AUROC, balanced
accuracy, threshold, and calibration metrics. NIH label variants such as
`Pleural_thickening` are normalized to the configured `Pleural Thickening`.

The inference prompt asks for **one correct category**, not for a visually
"dominant" category. Two strict full-vocabulary argmax metrics are reported:

- **Designated Primary Accuracy**: the argmax token exactly equals the
  training-designated first-positive primary code.
- **Any-Positive Accuracy**: the argmax token is in the label-token set and
  its ground-truth label is positive.

Both metrics are strict: an argmax outside the label-token set is incorrect
and is never replaced by sigmoid ranking. For datasets with official
no-positive test records (including ChestX-ray14), these argmax metrics are
defined only over records that have at least one positive label; no-positive
records remain in sigmoid-space metrics.

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Prepare PASCAL VOC 2007 data (standard multi-label protocol:
#    trainval->train, test->test). First extract:
#    tar -xf data/VOCtrainval_06-Nov-2007.tar -C data/
#    tar -xf data/VOCtest_06-Nov-2007.tar -C data/
vlm-multilabel convert --dataset voc \
    --voc-root data/VOCdevkit/VOC2007 --out data/voc2007 \
    --splits train:trainval,test:test

# 3. Train the main CE+BCE objective. Do not use the test split as validation.
#    Pass an empty val path when no held-out validation split exists.
vlm-multilabel train --model Qwen/Qwen3.5-4B \
    --dataset voc --train-data data/voc2007/train.jsonl \
    --val-data "" --output-dir checkpoints/voc_ce_bce \
    --mp-weight 0

# Optional: train the full diagnostic objective
vlm-multilabel train --model Qwen/Qwen3.5-4B \
    --dataset voc --train-data data/voc2007/train.jsonl \
    --val-data "" --output-dir checkpoints/voc_full

# 4. Evaluate with fp32 sigmoid and dual strict primary metrics
vlm-multilabel evaluate --model Qwen/Qwen3.5-4B \
    --adapters checkpoints/voc_ce_bce --dataset voc \
    --test-data data/voc2007/test.jsonl \
    --output results/voc_2007/fp32/first_positive/main/voc_eval.json
```

Evaluation upcasts label-token logits to float32 before `sigmoid`. Each
evaluation writes per-sample JSONL and reports a fixed threshold sweep, ECE,
Brier score, and binary NLL.

## Command layout

The canonical entry point is:

```bash
vlm-multilabel <command> [options...]
```

Supported commands are `convert`, `train`, `evaluate`, `train-aux-head`,
`evaluate-aux-head`, `posthoc-calibration`, `infer`, `resize-images`, and
`verify-theory`. For conversion, pass
`--dataset voc`, `--dataset chestxray14`, or `--dataset celeba`.

The legacy `scripts/*.py` entry points have been removed. Use the installed
`vlm-multilabel` command or `python -m vlm_multilabel.cli` instead.

## Project Structure

```text
vlm-multilabel/
  vlm_multilabel/
    cli.py                # Unified `vlm-multilabel` entry point
    commands/             # train, evaluate, infer, converters, diagnostics
    constants/            # Dataset label taxonomy and prompt templates
    data/                 # Dataset conversion and image utilities
    losses/               # CE + optional MP + BCE implementation
    runtime/              # VLM loading and ms-swift compatibility
    metrics.py            # Shared metric implementations
  docs/
    environment.md        # Tested package versions and reproduction notes
  results/
    */fp32/first_positive/main/       # Metric JSON files (main objective)
    */fp32/first_positive/per_sample/ # Per-sample diagnostics
    */fp32/first_positive/aux_head*/  # Aux-head control results
    */fp32/first_positive/ce_bce_auxhead/ # One-factor placement cell
    diagnostics/          # Theory-verification outputs
    posthoc_calibration.json, voc_bootstrap_ci.txt
  scripts/
    run_experiments.sh    # Full experiment matrix (staged, idempotent)
  data/                   # Generated local data files (gitignored)
```

## Datasets

| Dataset | Domain | Classes | Images | Label Density | Role |
|---------|--------|---------|--------|---------------|------|
| PASCAL VOC 2007 | General objects | 20 | 10K | Sparse (1.5/img) | Main ablation |
| ChestX-ray14 | Medical imaging | 14 | 112K (30K subsample) | Medium (2-3/img) | Hard-domain stress test |
| CelebA | Face attributes | 40 | 203K (30K subsample) | Dense (9/img) | Cross-domain validation |

For training/validation, record counts exclude zero-positive records skipped by
the first-positive converters. The ChestX-ray14 test record count includes all
official test records.

## Loss Components

### L_CE
Standard next-token cross-entropy on the assistant response sequence,
implemented by ms-swift's `Seq2SeqTrainer.compute_loss`.

### L_MP (optional set-level multi-positive loss)
MP uses a set-level margin-ranking implementation (internal code name:
`setrank`):

```text
L_MP = softplus(MP_MARGIN + logsumexp(invalid) - logsumexp(valid))
```

`MP_MARGIN` is fixed to `2.0`. It is optional and useful for diagnosis and
argmax-level analysis.

### L_BCE
Binary cross-entropy with logits on the `N` label-token logits at the first
label-code position, supervised by the per-sample `label_vector`. This is the
main method together with CE.

## Inference

At inference time, evaluation performs a single non-autoregressive forward
pass:

1. Extract full-vocabulary logits at the next-token position.
2. Score Designated Primary Accuracy and Any-Positive Accuracy strictly.
3. Extract the `N` label-token logits and upcast them to float32.
4. Apply sigmoid and compute mAP, balanced accuracy, AUROC, ECE, Brier, and
   binary NLL.
5. Write per-sample logits/probabilities for offline threshold and distribution
   diagnostics.

## Requirements

- Python >= 3.10
- ms-swift >= 4.0
- transformers >= 5.0
- torch >= 2.0
- See `requirements.txt` for the full list.

## Model weights

No trained checkpoints or LoRA adapters are shipped with this repository
(size). Reproducing the released results requires retraining from the configs
in `scripts/run_experiments.sh` (the cells are 1-2 epochs of LoRA
fine-tuning per seed on 4B-class VLMs). All per-sample dumps and evaluation
artifacts needed to verify the released results and statistics are included
under `results/`.

## License

Released under the [MIT License](LICENSE).
