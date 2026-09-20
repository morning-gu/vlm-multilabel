# Environment (tested versions)

The released results were produced on a single NVIDIA GeForce RTX 4090
(24 GB, driver 595.71.05, CUDA 12.8) with the following package versions.
The ranges in `requirements.txt` are the supported envelope; **pin to these
exact versions for a byte-comparable reproduction** — flash-attention and
batching can flip near-tie decisions on a frozen decoder, so the bf16/fp32
evaluation artifacts are precision- and operation-order-sensitive.

| Package | Tested version |
|---|---|
| Python | 3.12.3 |
| ms-swift | 4.5.3 |
| transformers | 5.16.1 |
| torch | 2.7.1 (CUDA) |
| torchvision | 0.22.1 |
| peft | 0.21.0 |
| accelerate | 1.11.0 |
| flash-attn | 2.8.3.post1 |
| trl | 0.23.1 |
| numpy | 2.2.6 |
| scipy | 1.16.3 |

Notes:

- Attention implementation: the released runs use `flash_attention_2`
  (override with `ATTN_IMPLEMENTATION=sdpa|eager` via
  `scripts/run_experiments.sh`). Flash-attention vs. sdpa can flip
  near-tie argmax decisions on a frozen decoder (≤ 11 of 4,952 images on
  VOC; see the per-sample dumps under `results/`).
- Training/evaluation entry points: `vlm-multilabel train` /
  `vlm-multilabel evaluate` (see README Quick Start and
  `scripts/run_experiments.sh` for the full matrix).
