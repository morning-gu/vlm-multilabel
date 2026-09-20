#!/usr/bin/env bash
# Run the full experiment matrix (first-positive protocol).
#
# Matrix:
#   A  VOC 2007      5 variants x 3 seeds (retrain, main table)
#   B  ChestX-ray14  zero-shot + CE-only/CE+BCE x seed 123 (adaptive; add seeds if collapse reproduces)
#   C  CelebA 30K    zero-shot + CE-only/CE+BCE x seed 123
#   D  Gemma3 VOC    zero-shot/CE-only/CE+BCE x seed 42 (architecture spot-check)
#   E  VOC aux-head  3 seeds (independent classification head + BCE control)
#   F  post-hoc calibration (offline, from per-sample logits)
#
# Usage:
#   bash scripts/run_experiments.sh [stage]
#   stage: all | voc | cxr | celeba | gemma3 | auxhead | newcell | posthoc
#
# All stages are idempotent: runs whose result JSON already exists are skipped.
# Override models/data via environment variables, e.g.:
#   MODEL_QWEN=Qwen/Qwen3.5-4B DATA_VOC=data/voc_2007 bash scripts/run_experiments.sh voc
# Override attention implementation (flash_attention_2 | sdpa | eager), e.g.:
#   ATTN_IMPLEMENTATION=sdpa bash scripts/run_experiments.sh voc

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_QWEN="${MODEL_QWEN:-Qwen/Qwen3.5-4B}"
MODEL_GEMMA="${MODEL_GEMMA:-google/gemma-3-4b-it}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

DATA_VOC="${DATA_VOC:-data/voc_2007}"
DATA_CXR="${DATA_CXR:-data/chestxray14_30k}"
DATA_CELEBA="${DATA_CELEBA:-data/celeba_30k}"

CKPT_ROOT="${CKPT_ROOT:-checkpoints}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

SEEDS_VOC=(42 123 2026)
SEED_DIAG=123
SEED_GEMMA=42
EPOCHS_VOC="${EPOCHS_VOC:-2}"
EPOCHS_CXR="${EPOCHS_CXR:-1}"
EPOCHS_CELEBA="${EPOCHS_CELEBA:-1}"

STAGE="${1:-all}"

# Resolve the CLI entry point; fall back to the module form when the console
# script is not on PATH (e.g. background shells that skip profile loading).
if command -v vlm-multilabel >/dev/null 2>&1; then
    VLM=(vlm-multilabel)
else
    VLM=(python -m vlm_multilabel.cli)
fi
echo "CLI entry: ${VLM[*]}"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== Experiment run started $(date) (log: $LOG_FILE) ==="
echo "Attention implementation: $ATTN_IMPLEMENTATION"

train_main() { # model dataset train_data variant seed ckpt
    local model="$1" dataset="$2" train="$3" variant="$4" seed="$5" ckpt="$6"
    local flags=()
    local epochs=2
    case "$dataset" in
        voc) epochs="$EPOCHS_VOC" ;;
        chestxray14) epochs="$EPOCHS_CXR" ;;
        celeba) epochs="$EPOCHS_CELEBA" ;;
    esac
    case "$variant" in
        ce_only) flags=(--bce-weight 0 --mp-weight 0) ;;
        ce_mp)   flags=(--bce-weight 0 --mp-weight 1.0) ;;
        ce_bce)  flags=(--mp-weight 0) ;;
        full)    flags=() ;;
        *) echo "unknown variant: $variant"; exit 1 ;;
    esac
    echo "--- train: dataset=$dataset variant=$variant seed=$seed model=$model epochs=$epochs checkpoint=$ckpt"
    "${VLM[@]}" train \
        --model "$model" \
        --dataset "$dataset" \
        --train-data "$train" \
        --val-data "" \
        --output-dir "$ckpt" \
        --seed "$seed" \
        --num-train-epochs "$epochs" \
        --attn-implementation "$ATTN_IMPLEMENTATION" \
        "${flags[@]}"
}

eval_main() { # model dataset test_data ckpt out_json per_sample_json
    local model="$1" dataset="$2" test="$3" ckpt="$4" out_json="$5" per_sample="$6"
    mkdir -p "$(dirname "$out_json")" "$(dirname "$per_sample")"
    echo "--- eval: dataset=$dataset checkpoint=${ckpt:-none}"
    "${VLM[@]}" evaluate \
        --model "$model" \
        --dataset "$dataset" \
        --test-data "$test" \
        --adapters "$ckpt" \
        --output "$out_json" \
        --per-sample-output "$per_sample" \
        --attn-implementation "$ATTN_IMPLEMENTATION"
}

run_main_variant() { # model dataset data_dir result_dir_name variant seed
    local model="$1" dataset="$2" data_dir="$3" result_dir="$4" variant="$5" seed="$6"
    local ckpt="$CKPT_ROOT/${result_dir}_${variant}_seed_${seed}"
    local out_json="$RESULTS_ROOT/$result_dir/fp32/first_positive/main/${result_dir}_${variant}_seed_${seed}.json"
    local per_sample="$RESULTS_ROOT/$result_dir/fp32/first_positive/per_sample/${result_dir}_${variant}_seed_${seed}.jsonl"
    if [[ -f "$out_json" ]]; then
        echo "skip $out_json (exists)"
        return 0
    fi
    train_main "$model" "$dataset" "$data_dir/train.jsonl" "$variant" "$seed" "$ckpt"
    eval_main "$model" "$dataset" "$data_dir/test.jsonl" "$ckpt/last" "$out_json" "$per_sample"
}

run_zero_shot() { # model dataset data_dir result_dir_name
    local model="$1" dataset="$2" data_dir="$3" result_dir="$4"
    local out_json="$RESULTS_ROOT/$result_dir/fp32/first_positive/main/${result_dir}_zero_shot.json"
    local per_sample="$RESULTS_ROOT/$result_dir/fp32/first_positive/per_sample/${result_dir}_zero_shot.jsonl"
    if [[ -f "$out_json" ]]; then
        echo "skip $out_json (exists)"
        return 0
    fi
    eval_main "$model" "$dataset" "$data_dir/test.jsonl" "" "$out_json" "$per_sample"
}

run_aux_head() { # dataset data_dir result_dir_name seed
    local dataset="$1" data_dir="$2" result_dir="$3" seed="$4"
    local ckpt="$CKPT_ROOT/${result_dir}_aux_head_seed_${seed}"
    local out_json="$RESULTS_ROOT/$result_dir/fp32/first_positive/aux_head/${result_dir}_aux_head_seed_${seed}.json"
    local per_sample="$RESULTS_ROOT/$result_dir/fp32/first_positive/aux_head/${result_dir}_aux_head_seed_${seed}.jsonl"
    if [[ -f "$out_json" ]]; then
        echo "skip $out_json (exists)"
        return 0
    fi
    echo "--- train-aux-head: dataset=$dataset seed=$seed"
    "${VLM[@]}" train-aux-head \
        --model "$MODEL_QWEN" \
        --dataset "$dataset" \
        --train-data "$data_dir/train.jsonl" \
        --output-dir "$ckpt" \
        --seed "$seed" \
        --attn-implementation "$ATTN_IMPLEMENTATION"
    mkdir -p "$(dirname "$out_json")" "$(dirname "$per_sample")"
    echo "--- eval-aux-head: dataset=$dataset seed=$seed"
    "${VLM[@]}" evaluate-aux-head \
        --model "$MODEL_QWEN" \
        --adapters "$ckpt" \
        --aux-head "$ckpt/aux_head.pt" \
        --dataset "$dataset" \
        --test-data "$data_dir/test.jsonl" \
        --output "$out_json" \
        --per-sample-output "$per_sample" \
        --attn-implementation "$ATTN_IMPLEMENTATION"
}

run_posthoc() {
    local out_json="$RESULTS_ROOT/posthoc_calibration.json"
    local patterns=()
    for dir in voc_2007 chestxray14_30k celeba_30k; do
        local pattern="$RESULTS_ROOT/$dir/fp32/first_positive/per_sample/*.jsonl"
        if compgen -G "$pattern" > /dev/null; then
            patterns+=("$pattern")
        fi
    done
    if [[ ${#patterns[@]} -eq 0 ]]; then
        echo "no per-sample files found; skipping post-hoc calibration"
        return 0
    fi
    echo "--- post-hoc calibration over ${#patterns[@]} pattern(s)"
    "${VLM[@]}" posthoc-calibration \
        --per-sample "${patterns[@]}" \
        --output "$out_json"
}

stage_voc() {
    echo "=== Stage A: VOC 2007, 5 variants x 3 seeds ==="
    run_zero_shot "$MODEL_QWEN" voc "$DATA_VOC" voc_2007
    for seed in "${SEEDS_VOC[@]}"; do
        for variant in ce_only ce_mp ce_bce full; do
            run_main_variant "$MODEL_QWEN" voc "$DATA_VOC" voc_2007 "$variant" "$seed"
        done
    done
}

stage_cxr() {
    echo "=== Stage B: ChestX-ray14, adaptive seeds ==="
    run_zero_shot "$MODEL_QWEN" chestxray14 "$DATA_CXR" chestxray14_30k
    for variant in ce_only ce_bce; do
        run_main_variant "$MODEL_QWEN" chestxray14 "$DATA_CXR" chestxray14_30k "$variant" "$SEED_DIAG"
    done
    echo "NOTE: check whether the CXR collapse reproduces (AnyPos Acc drop >= 10 pt)."
    echo "      If it does, add seeds 42/2026 for ce_only+ce_bce (matrix item B2)."
}

stage_celeba() {
    echo "=== Stage C: CelebA 30K, 1 seed ==="
    run_zero_shot "$MODEL_QWEN" celeba "$DATA_CELEBA" celeba_30k
    for variant in ce_only ce_bce; do
        run_main_variant "$MODEL_QWEN" celeba "$DATA_CELEBA" celeba_30k "$variant" "$SEED_DIAG"
    done
}

stage_gemma3() {
    echo "=== Stage D: Gemma3 on VOC, 1 seed ==="
    run_zero_shot "$MODEL_GEMMA" voc "$DATA_VOC" voc_2007_gemma3
    for variant in ce_only ce_bce; do
        run_main_variant "$MODEL_GEMMA" voc "$DATA_VOC" voc_2007_gemma3 "$variant" "$SEED_GEMMA"
    done
}

stage_auxhead() {
    echo "=== Stage E: VOC aux-head control, 3 seeds ==="
    for seed in "${SEEDS_VOC[@]}"; do
        run_aux_head voc "$DATA_VOC" voc_2007 "$seed"
    done
}

# One-factor placement cell: CE(decoder) + BCE(aux-head).
# Stage 1 reuses the existing ce_only LoRA checkpoints (the decoder CE half,
# frozen); stage 2 fits only the aux head with BCE on their last-token hidden
# states. The decoder generative space is by construction identical to the
# ce_only cell, so this stage measures the sigmoid half and dumps the decoder
# argmax fields reported as n/a in the released evaluation artifacts.
run_new_cell() { # seed
    local seed="$1"
    local ce_ckpt="$CKPT_ROOT/voc_2007_ce_only_seed_${seed}/last"
    local new_ckpt="$CKPT_ROOT/voc_2007_ce_bce_auxhead_seed_${seed}"
    local out_dir="$RESULTS_ROOT/voc_2007/fp32/first_positive/ce_bce_auxhead"
    local out_json="$out_dir/voc_2007_ce_bce_auxhead_seed_${seed}.json"
    local per_sample="$out_dir/voc_2007_ce_bce_auxhead_seed_${seed}.jsonl"
    if [[ ! -d "$ce_ckpt" ]]; then
        echo "missing ce_only checkpoint $ce_ckpt; run the voc stage first" >&2
        exit 1
    fi
    if [[ ! -f "$new_ckpt/aux_head.pt" ]]; then
        echo "--- train new-cell aux head: seed=$seed (frozen CE-only backbone)"
        "${VLM[@]}" train-aux-head \
            --model "$MODEL_QWEN" \
            --dataset voc \
            --train-data "$DATA_VOC/train.jsonl" \
            --adapters "$ce_ckpt" \
            --freeze-backbone \
            --output-dir "$new_ckpt" \
            --seed "$seed" \
            --attn-implementation "$ATTN_IMPLEMENTATION"
    fi
    mkdir -p "$out_dir"
    echo "--- eval new cell: seed=$seed"
    "${VLM[@]}" evaluate-aux-head \
        --model "$MODEL_QWEN" \
        --adapters "$ce_ckpt" \
        --aux-head "$new_ckpt/aux_head.pt" \
        --dataset voc \
        --test-data "$DATA_VOC/test.jsonl" \
        --output "$out_json" \
        --per-sample-output "$per_sample" \
        --dump-decoder-argmax \
        --attn-implementation "$ATTN_IMPLEMENTATION"
}

# Re-inference pass over the existing aux-head-only checkpoints: adds the
# decoder strict-argmax fields (untuned-backbone generative half) that the
# first auxhead stage could not report.
redump_aux_head() { # seed
    local seed="$1"
    local ckpt="$CKPT_ROOT/voc_2007_aux_head_seed_${seed}"
    local out_dir="$RESULTS_ROOT/voc_2007/fp32/first_positive/aux_head_argmax"
    local out_json="$out_dir/voc_2007_aux_head_seed_${seed}.json"
    local per_sample="$out_dir/voc_2007_aux_head_seed_${seed}.jsonl"
    if [[ ! -f "$ckpt/aux_head.pt" ]]; then
        echo "missing aux-head checkpoint $ckpt; run the auxhead stage first" >&2
        exit 1
    fi
    mkdir -p "$out_dir"
    echo "--- re-eval aux-head with decoder argmax: seed=$seed"
    "${VLM[@]}" evaluate-aux-head \
        --model "$MODEL_QWEN" \
        --adapters "$ckpt" \
        --aux-head "$ckpt/aux_head.pt" \
        --dataset voc \
        --test-data "$DATA_VOC/test.jsonl" \
        --output "$out_json" \
        --per-sample-output "$per_sample" \
        --dump-decoder-argmax \
        --attn-implementation "$ATTN_IMPLEMENTATION"
}

stage_newcell() {
    echo "=== Stage G: VOC one-factor placement cell + aux-head re-dump, 3 seeds ==="
    for seed in "${SEEDS_VOC[@]}"; do
        run_new_cell "$seed"
        redump_aux_head "$seed"
    done
}

stage_posthoc() {
    echo "=== Stage F: offline post-hoc calibration ==="
    run_posthoc
}

case "$STAGE" in
    voc)     stage_voc ;;
    cxr)     stage_cxr ;;
    celeba)  stage_celeba ;;
    gemma3)  stage_gemma3 ;;
    auxhead) stage_auxhead ;;
    newcell) stage_newcell ;;
    posthoc) stage_posthoc ;;
    all)
        stage_voc
        stage_cxr
        stage_celeba
        stage_gemma3
        stage_auxhead
        stage_newcell
        stage_posthoc
        ;;
    *) echo "unknown stage: $STAGE (use all|voc|cxr|celeba|gemma3|auxhead|newcell|posthoc)"; exit 1 ;;
esac

echo "=== done $(date). Results under $RESULTS_ROOT/*/fp32/first_positive/ ==="
