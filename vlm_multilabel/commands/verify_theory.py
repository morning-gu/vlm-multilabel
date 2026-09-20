"""Verify the self-extinguishing theorem by comparing CE-only and CE+MP checkpoints.

Four experiments to distinguish the three hypotheses in docs/theoretical_analysis.md:

  A. Weight L2 distance: if ||W_ce_mp - W_ce_only|| == 0, MP had zero gradient
     effect (bf16 underflow). If > 0, weights diverged.
  B. Raw logit comparison: compare label-token logits (not AP) at full precision
     between CE-only and CE+MP checkpoints. Also compute per-class AP to check
     if AP is identical despite different logits.
  C. Per-sample sigma logging: log sigmoid(margin + LSE(invalid) - LSE(valid))
     during eval to see how often sigma is negligible (verifies Theorem 1).
  D. Same-checkpoint determinism test: run inference TWICE on the same checkpoint
     and compare logits. This establishes a non-determinism noise floor.

  Sigmoid computation matches eval_multilabel.py exactly:
    - sigmoid computed in model dtype (bf16) on GPU, THEN upcast to float32
    - records sorted by image file size (reduces padding, matches eval batching)

  Use --sigmoid-float32 to compute sigmoid in float32 on CPU instead (the old
  behavior that produced different AP from eval_multilabel.py).  This lets you
  compare both modes to confirm the precision difference is the root cause.

Usage:
  vlm-multilabel verify-theory \
      --model Qwen/Qwen3.5-4B \
      --ce-only-checkpoint checkpoints/ce_only \
      --ce-mp-checkpoint checkpoints/ce_mp_setrank \
      --dataset voc --test-data data/voc/test.jsonl \
      --experiments A,B,C,D

  To confirm sigmoid precision is the root cause of the AP discrepancy:
  vlm-multilabel verify-theory ... --experiments B --sigmoid-float32
  # With --sigmoid-float32:  AP differs (float32 sigmoid exposes logit diffs)
  # Without (default):        AP matches (bf16 sigmoid masks logit diffs, same as eval)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from vlm_multilabel.constants.base import load_config


# ---------------------------------------------------------------------------
# Reusable inference helpers
# ---------------------------------------------------------------------------

def _load_model(model_id, ckpt_path, attn_impl):
    """Load a VLM with LoRA adapters and return (model, template, tokenizer)."""
    from transformers import AutoModelForImageTextToText
    from swift.tuners import Swift
    from swift import get_processor as _sw_get_processor, get_template as _sw_get_template

    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto",
        trust_remote_code=True, attn_implementation=attn_impl)
    if ckpt_path:
        model = Swift.from_pretrained(model, ckpt_path)
    model.eval()

    sw_processor = _sw_get_processor(model_id)
    template = _sw_get_template(sw_processor, enable_thinking=False,
                                padding_side="left")
    if getattr(template, "use_model", False):
        template.model = model
    tokenizer = sw_processor.tokenizer
    return model, template, tokenizer


def _maybe_sort_by_size(records):
    """Sort records by image file size (matches eval_multilabel.py default)."""
    records.sort(key=lambda r: os.path.getsize(r["images"][0]))
    return records


def _run_inference(model, template, config, records, id_tensor,
                   sys_prompt, batch_size=16, sigmoid_float32=False):
    """Run prompt-only inference.

    Returns (logits_float32[B,N], argmax_tokens[B], sigmoid_probs[B,N]).

    By default, sigmoid is computed in the model's native dtype (bf16) on GPU,
    then upcast to float32 --- matching eval_multilabel.py line 441:
        probs = torch.sigmoid(label_logits).float().cpu().numpy()

    With sigmoid_float32=True, logits are first upcast to float32 and sigmoid
    is computed in float32 on CPU.  This is more precise but DOES NOT match
    eval_multilabel.py; it exposes logit differences that bf16 sigmoid masks.
    """
    import torch
    from PIL import Image
    from swift import InferRequest

    logits_list = []
    argmax_list = []
    probs_list = []

    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        images = [Image.open(r["images"][0]).convert("RGB") for r in batch]
        reqs = [InferRequest(
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": r["messages"][1]["content"]}],
            images=[img]) for r, img in zip(batch, images)]
        encoded = [template.encode(r, return_template_inputs=True) for r in reqs]
        for e in encoded:
            e.pop("template_inputs", None)
        inputs = template.data_collator(encoded)
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        _, inputs = template.pre_forward_hook(model, None, inputs)

        with torch.inference_mode():
            outputs = model(**inputs, use_cache=False, num_logits_to_keep=1)
            last_logits = outputs.logits[:, -1, :]
            label_logits = last_logits[:, id_tensor]

            # Logits as float32 for cross-checkpoint comparison (always)
            logits_list.append(label_logits.float().cpu())

            # Argmax over full vocab (always)
            argmax_list.append(last_logits.argmax(dim=-1).cpu())

            # Sigmoid: match eval_multilabel.py (bf16 on GPU) or use float32
            if sigmoid_float32:
                # Old behavior: float32 sigmoid on CPU (more precise, NOT matching eval)
                probs_list.append(torch.sigmoid(
                    label_logits.float().cpu()))
            else:
                # eval_match: sigmoid in model dtype (bf16) on GPU, then float32
                # This is EXACTLY what eval_multilabel.py does:
                #   probs = torch.sigmoid(label_logits).float().cpu().numpy()
                probs_list.append(
                    torch.sigmoid(label_logits).float().cpu())

        torch.cuda.empty_cache()

    all_logits = torch.cat(logits_list, dim=0)
    all_argmax = torch.cat(argmax_list, dim=0)
    all_probs = torch.cat(probs_list, dim=0).numpy()
    return all_logits, all_argmax, all_probs


def _compute_ap(y_true, probs, n_classes):
    """Compute per-class AP from probabilities. Returns (mean_ap, per_class_aps)."""
    from sklearn.metrics import average_precision_score
    aps = []
    for c in range(n_classes):
        if y_true[:, c].sum() == 0:
            continue
        aps.append(float(average_precision_score(y_true[:, c], probs[:, c])))
    return float(np.mean(aps)) if aps else 0.0, aps


def _build_y_true(records, n_classes):
    """Build ground-truth label matrix from records."""
    y_true = np.zeros((len(records), n_classes), dtype=np.float32)
    for i, r in enumerate(records):
        lv = r.get("label_vector", [])
        for j, v in enumerate(lv):
            if v and j < n_classes:
                y_true[i, j] = 1.0
    return y_true


def _load_test_records(test_data, n=None, sort_by_size=True):
    """Load test JSONL records, optionally truncated and sorted by file size."""
    with open(test_data, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if n:
        records = records[:n]
    if sort_by_size:
        records = _maybe_sort_by_size(records)
    return records


def _setup_model_and_template(model_id, ckpt_path, dataset, attn_impl):
    """Load model, template, config, and derived objects. Returns a dict."""
    import torch

    config = load_config(dataset)
    model, template, tokenizer = _load_model(model_id, ckpt_path, attn_impl)
    label_token_ids = config.get_label_token_ids(tokenizer)
    id_list = list(label_token_ids.values())
    id_tensor = torch.tensor(id_list, device=model.device, dtype=torch.long)
    sys_prompt = config.render_prompt()
    return {
        "model": model, "template": template, "config": config,
        "id_list": id_list, "id_tensor": id_tensor, "sys_prompt": sys_prompt,
        "n_classes": len(id_list),
    }


# ---------------------------------------------------------------------------
# Experiment A: Weight L2 distance
# ---------------------------------------------------------------------------

def compare_weights(ce_only_ckpt, ce_mp_ckpt):
    """Experiment A: compute L2 distance between checkpoint weights."""
    import torch

    print("\n=== Experiment A: Weight L2 Distance ===")

    sd_ce = torch.load(Path(ce_only_ckpt) / "adapter_model.safetensors",
                       weights_only=False)
    sd_mp = torch.load(Path(ce_mp_ckpt) / "adapter_model.safetensors",
                       weights_only=False)

    total_l2 = 0.0
    total_abs_max = 0.0
    n_params = 0
    n_diff = 0

    for key in sd_ce:
        if key not in sd_mp:
            print(f"  WARNING: key {key} missing in CE+MP checkpoint")
            continue
        diff = (sd_mp[key].float() - sd_ce[key].float())
        l2 = diff.norm().item()
        abs_max = diff.abs().max().item()
        total_l2 += l2 ** 2
        total_abs_max = max(total_abs_max, abs_max)
        n_params += diff.numel()
        if l2 > 0:
            n_diff += 1

    total_l2 = total_l2 ** 0.5

    print(f"  Total params compared: {n_params}")
    print(f"  Params with nonzero diff: {n_diff}")
    print(f"  L2 distance: {total_l2:.10f}")
    print(f"  Max abs diff: {total_abs_max:.10f}")

    if total_l2 == 0.0:
        print("  => Hypothesis A CONFIRMED: weights are bit-for-bit identical")
        print("     MP gradient was exactly zero (bf16 underflow likely)")
    else:
        print(f"  => Hypothesis B/C: weights differ (L2={total_l2:.2e})")
        print("     MP had nonzero gradient effect")

    return {"l2_distance": total_l2, "max_abs_diff": total_abs_max,
            "n_params": n_params, "n_diff": n_diff}


# ---------------------------------------------------------------------------
# Experiment B: Cross-checkpoint logit comparison
# ---------------------------------------------------------------------------

def compare_logits(model_id, ce_only_ckpt, ce_mp_ckpt, dataset, test_data,
                   batch_size=16, attn_impl="flash_attention_2",
                   sort_by_size=True, sigmoid_float32=False):
    """Experiment B: compare raw label-token logits between CE-only and CE+MP.

    Sigmoid precision and sorting match eval_multilabel.py by default.
    Set sigmoid_float32=True to use float32 sigmoid (more precise, exposes
    logit differences that bf16 sigmoid masks).
    """
    import torch

    sigmoid_desc = "float32 (CPU, precise)" if sigmoid_float32 else "bf16 (GPU, matches eval)"
    print("\n=== Experiment B: Cross-Checkpoint Logit Comparison ===")
    print(f"  Sigmoid mode: {sigmoid_desc}")
    print(f"  Sort by size: {sort_by_size}")

    config = load_config(dataset)
    records = _load_test_records(test_data, n=min(200, len(_load_test_records(
        test_data, sort_by_size=False))), sort_by_size=sort_by_size)
    n_compare = len(records)
    print(f"  Comparing logits on {n_compare} images")

    y_true = _build_y_true(records, len(config.label_codes))

    results = {}
    all_logits = {}
    all_argmax = {}
    all_probs = {}

    for ckpt_name, ckpt_path in [("CE-only", ce_only_ckpt), ("CE+MP", ce_mp_ckpt)]:
        print(f"  Loading {ckpt_name}: {ckpt_path}")
        ctx = _setup_model_and_template(model_id, ckpt_path, dataset, attn_impl)
        logits, argmax, probs = _run_inference(
            ctx["model"], ctx["template"], config, records,
            ctx["id_tensor"], ctx["sys_prompt"], batch_size,
            sigmoid_float32=sigmoid_float32)
        all_logits[ckpt_name] = logits
        all_argmax[ckpt_name] = argmax
        all_probs[ckpt_name] = probs
        del ctx["model"]
        torch.cuda.empty_cache()

    logit_diff = (all_logits["CE+MP"] - all_logits["CE-only"])
    abs_diff = logit_diff.abs()

    mean_ap_ce, ap_ce = _compute_ap(y_true, all_probs["CE-only"], len(config.label_codes))
    mean_ap_mp, ap_mp = _compute_ap(y_true, all_probs["CE+MP"], len(config.label_codes))
    ap_max_diff = max(abs(a - b) for a, b in zip(ap_ce, ap_mp)) if ap_ce else 0.0
    ap_match = ap_max_diff == 0.0

    argmax_match = (all_argmax["CE-only"] == all_argmax["CE+MP"]).sum().item()

    print(f"  Logit diff: max={abs_diff.max():.6f}, mean={abs_diff.mean():.6f}, "
          f"L2={logit_diff.norm():.6f}")
    print(f"  Argmax match: {argmax_match}/{n_compare} ({argmax_match/n_compare:.4f})")
    print(f"  AP (subset): CE-only={mean_ap_ce:.6f}, CE+MP={mean_ap_mp:.6f}, "
          f"max per-class diff={ap_max_diff:.6f}")
    if ap_match:
        print("  => AP identical on subset (bf16 sigmoid masked logit diffs, "
              "matching eval_multilabel.py)")
    else:
        print("  => AP differs on subset (ranking changed)")

    results.update({
        "sigmoid_mode": sigmoid_desc,
        "sort_by_size": sort_by_size,
        "logit_max_diff": float(abs_diff.max()),
        "logit_mean_diff": float(abs_diff.mean()),
        "logit_l2_diff": float(logit_diff.norm()),
        "argmax_match_rate": float(argmax_match / n_compare),
        "ap_ce_only": mean_ap_ce,
        "ap_ce_mp": mean_ap_mp,
        "ap_max_per_class_diff": float(ap_max_diff),
        "ap_match": bool(ap_match),
        "per_class_ap_ce": ap_ce,
        "per_class_ap_mp": ap_mp,
    })
    return results


# ---------------------------------------------------------------------------
# Experiment C: Sigma statistics
# ---------------------------------------------------------------------------

def log_sigma_stats(model_id, ckpt_path, dataset, test_data,
                    batch_size=16, margin=2.0, attn_impl="flash_attention_2",
                    sort_by_size=True):
    """Experiment C: compute sigma on test set to verify Theorem 1."""
    import torch
    from PIL import Image
    from swift import InferRequest

    print("\n=== Experiment C: Sigma (MP Activation) Statistics ===")

    records = _load_test_records(test_data, n=min(500, len(_load_test_records(
        test_data, sort_by_size=False))), sort_by_size=sort_by_size)
    n_eval = len(records)
    print(f"  Computing sigma on {n_eval} images")

    ctx = _setup_model_and_template(model_id, ckpt_path, dataset, attn_impl)
    model = ctx["model"]
    template = ctx["template"]
    id_tensor = ctx["id_tensor"]
    id_list = ctx["id_list"]
    sys_prompt = ctx["sys_prompt"]
    N = len(id_list)

    sigmas = []

    for i in range(0, n_eval, batch_size):
        batch = records[i:i+batch_size]
        images = [Image.open(r["images"][0]).convert("RGB") for r in batch]
        reqs = [InferRequest(
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": r["messages"][1]["content"]}],
            images=[img]) for r, img in zip(batch, images)]
        encoded = [template.encode(r, return_template_inputs=True) for r in reqs]
        for e in encoded:
            e.pop("template_inputs", None)
        inputs = template.data_collator(encoded)
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        _, inputs = template.pre_forward_hook(model, None, inputs)

        with torch.inference_mode():
            outputs = model(**inputs, use_cache=False, num_logits_to_keep=1)
            last_logits = outputs.logits[:, -1, :]
            label_logits = last_logits[:, id_tensor]

            for b in range(len(batch)):
                lv = batch[b].get("label_vector", [])
                y = torch.tensor(lv, device=model.device, dtype=torch.float32)
                valid_mask = y == 1
                n_valid = valid_mask.sum().item()
                if n_valid < 2 or n_valid >= N:
                    continue

                z = label_logits[b]
                masked_valid = z.masked_fill(~valid_mask, float("-inf"))
                masked_invalid = z.masked_fill(valid_mask, float("-inf"))
                lse_valid = torch.logsumexp(masked_valid, dim=0)
                lse_invalid = torch.logsumexp(masked_invalid, dim=0)
                sigma = torch.sigmoid(margin + lse_invalid - lse_valid)
                sigmas.append(sigma.item())

        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()

    sigmas = np.array(sigmas)
    print(f"  Multi-label samples: {len(sigmas)}")
    print(f"  sigma stats: min={sigmas.min():.6f}, max={sigmas.max():.6f}, "
          f"mean={sigmas.mean():.6f}, median={np.median(sigmas):.6f}")
    print(f"  sigma < 1e-3: {(sigmas < 1e-3).sum()}/{len(sigmas)} "
          f"({(sigmas < 1e-3).mean()*100:.1f}%)")
    print(f"  sigma < 1e-1: {(sigmas < 1e-1).sum()}/{len(sigmas)} "
          f"({(sigmas < 1e-1).mean()*100:.1f}%)")

    return {"sigma_mean": float(sigmas.mean()),
            "sigma_median": float(np.median(sigmas)),
            "sigma_min": float(sigmas.min()),
            "sigma_max": float(sigmas.max()),
            "frac_sigma_lt_1e3": float((sigmas < 1e-3).mean()),
            "frac_sigma_lt_1e1": float((sigmas < 1e-1).mean())}


# ---------------------------------------------------------------------------
# Experiment D: Same-checkpoint determinism test
# ---------------------------------------------------------------------------

def test_determinism(model_id, ce_only_ckpt, ce_mp_ckpt, dataset, test_data,
                     batch_size=16, attn_impl="flash_attention_2",
                     sort_by_size=True, sigmoid_float32=False):
    """Experiment D: run inference TWICE on the same checkpoint and compare."""
    import torch

    sigmoid_desc = "float32 (CPU, precise)" if sigmoid_float32 else "bf16 (GPU, matches eval)"
    print("\n=== Experiment D: Same-Checkpoint Determinism Test ===")
    print(f"  Attention impl: {attn_impl}")
    print(f"  Sigmoid mode: {sigmoid_desc}")
    print(f"  Sort by size: {sort_by_size}")
    print("  Strategy: load checkpoint once, run inference twice, compare")

    config = load_config(dataset)
    records = _load_test_records(test_data, n=min(200, len(_load_test_records(
        test_data, sort_by_size=False))), sort_by_size=sort_by_size)
    n_compare = len(records)
    print(f"  Comparing on {n_compare} images")

    y_true = _build_y_true(records, len(config.label_codes))

    results = {}
    summary = {}

    for ckpt_name, ckpt_path in [("CE-only", ce_only_ckpt),
                                  ("CE+MP", ce_mp_ckpt)]:
        print(f"\n  --- {ckpt_name} ({ckpt_path}) ---")
        ctx = _setup_model_and_template(model_id, ckpt_path, dataset, attn_impl)

        print("  Run 1 ...")
        logits_r1, argmax_r1, probs_r1 = _run_inference(
            ctx["model"], ctx["template"], config, records,
            ctx["id_tensor"], ctx["sys_prompt"], batch_size,
            sigmoid_float32=sigmoid_float32)

        print("  Run 2 ...")
        logits_r2, argmax_r2, probs_r2 = _run_inference(
            ctx["model"], ctx["template"], config, records,
            ctx["id_tensor"], ctx["sys_prompt"], batch_size,
            sigmoid_float32=sigmoid_float32)

        del ctx["model"]
        torch.cuda.empty_cache()

        diff = (logits_r2 - logits_r1).abs()
        argmax_match = (argmax_r1 == argmax_r2).sum().item()

        ap_r1, _ = _compute_ap(y_true, probs_r1, len(config.label_codes))
        ap_r2, _ = _compute_ap(y_true, probs_r2, len(config.label_codes))

        run_result = {
            "logit_max_diff": float(diff.max()),
            "logit_mean_diff": float(diff.mean()),
            "logit_l2_diff": float(diff.norm()),
            "argmax_match_rate": float(argmax_match / n_compare),
            "ap_run1": ap_r1,
            "ap_run2": ap_r2,
            "ap_match": bool(ap_r1 == ap_r2),
        }
        results[ckpt_name] = run_result

        print(f"  Logit diff:  max={diff.max():.6f}, mean={diff.mean():.6f}, "
              f"L2={diff.norm():.6f}")
        print(f"  Argmax match: {argmax_match}/{n_compare} "
              f"({argmax_match/n_compare:.4f})")
        print(f"  AP: run1={ap_r1:.6f}, run2={ap_r2:.6f}, "
              f"match={ap_r1 == ap_r2}")

        if diff.max() == 0.0:
            print(f"  => {ckpt_name} is DETERMINISTIC: two runs produce identical logits")
        else:
            print(f"  => {ckpt_name} is NON-DETERMINISTIC: max logit diff={diff.max():.2e}")

    print("\n  --- Noise Floor vs Cross-Checkpoint Summary ---")
    for name in ["CE-only", "CE+MP"]:
        d = results[name]
        print(f"  {name} same-ckpt noise: max_diff={d['logit_max_diff']:.6f}, "
              f"argmax_match={d['argmax_match_rate']:.4f}")

    noise_ce = results["CE-only"]["logit_max_diff"]
    noise_mp = results["CE+MP"]["logit_max_diff"]
    noise_max = max(noise_ce, noise_mp)

    summary["noise_floor_max"] = noise_max
    summary["noise_floor_ce_only"] = noise_ce
    summary["noise_floor_ce_mp"] = noise_mp
    summary["deterministic"] = (noise_max == 0.0)
    results["summary"] = summary

    if noise_max == 0.0:
        print("\n  => CONCLUSION: Inference is deterministic (noise floor = 0)")
        print("     All Exp B logit differences are from WEIGHT changes")
    else:
        print(f"\n  => CONCLUSION: Non-determinism noise floor = {noise_max:.2e}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="base model id/path")
    ap.add_argument("--ce-only-checkpoint", required=True, help="CE-only LoRA dir")
    ap.add_argument("--ce-mp-checkpoint", required=True, help="CE+MP setrank LoRA dir")
    ap.add_argument("--dataset", required=True, choices=["voc", "chestxray14", "celeba"])
    ap.add_argument("--test-data", required=True, help="test jsonl path")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--attn-implementation", default="flash_attention_2",
                    help="flash_attention_2 (default) | sdpa | eager.")
    ap.add_argument("--experiments", default="A,B,C,D",
                    help="comma-separated: A=weights, B=logits, C=sigma, D=determinism")
    ap.add_argument("--sort-by-size", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="sort images by file size to match eval_multilabel.py "
                         "batching (default: on)")
    ap.add_argument("--sigmoid-float32", action="store_true", default=False,
                    help="compute sigmoid in float32 on CPU instead of bf16 on GPU. "
                         "float32 is more precise but does NOT match eval_multilabel.py. "
                         "Use this to confirm that bf16 sigmoid precision is the root "
                         "cause of AP differences between verify_theory.py and eval.")
    ap.add_argument("--output", default="results/diagnostics/theory/theory_verification_float32.json",
                    help="output JSON path")
    a = ap.parse_args(argv)

    exps = set(a.experiments.upper().split(","))
    results = {}

    if "A" in exps:
        results["weight_comparison"] = compare_weights(
            a.ce_only_checkpoint, a.ce_mp_checkpoint)

    if "B" in exps:
        results["logit_comparison"] = compare_logits(
            a.model, a.ce_only_checkpoint, a.ce_mp_checkpoint,
            a.dataset, a.test_data, a.batch_size, a.attn_implementation,
            sort_by_size=a.sort_by_size, sigmoid_float32=a.sigmoid_float32)

    if "C" in exps:
        results["sigma_stats"] = log_sigma_stats(
            a.model, a.ce_mp_checkpoint, a.dataset, a.test_data,
            a.batch_size, attn_impl=a.attn_implementation,
            sort_by_size=a.sort_by_size)

    if "D" in exps:
        results["determinism_test"] = test_determinism(
            a.model, a.ce_only_checkpoint, a.ce_mp_checkpoint,
            a.dataset, a.test_data, a.batch_size, a.attn_implementation,
            sort_by_size=a.sort_by_size, sigmoid_float32=a.sigmoid_float32)

    out_path = Path(a.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()