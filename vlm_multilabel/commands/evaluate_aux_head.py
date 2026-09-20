"""Evaluate the aux-head control baseline on a multi-label test split.

Single forward pass per batch: sigmoid of the aux-head logits (float32) feeds
mAP, balanced accuracy, AUROC, ECE, Brier, NLL, threshold sweep, and
per-sample JSONL output. The aux-head readout has no generative argmax, so
the two strict primary-argmax metrics are reported as null.

With --dump-decoder-argmax, the same last-token hidden state is additionally
projected through the language-model head to recover the decoder's next-token
distribution at exactly the aux-head readout position: full-vocabulary
argmax (restricted to the label set for the strict metrics) and the per-label
decoder logits are reported alongside the aux-head metrics, and the
per-sample JSONL gains the decoder argmax fields used by the main-method
evaluation.
"""
from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from vlm_multilabel.commands.evaluate import _encode_batch
from vlm_multilabel.constants.base import load_config
from vlm_multilabel.data.conversion import validate_first_positive_records
from vlm_multilabel.losses.aux_head import AuxHead, gather_last_hidden
from vlm_multilabel.runtime.model import keep_last_logits_kwargs
from vlm_multilabel.metrics import (
    compute_accuracy,
    compute_auc,
    compute_brier,
    compute_ece,
    compute_map,
    compute_nll,
    compute_threshold_sweep,
)
from vlm_multilabel.runtime.model import build_label_matrix, load_jsonl, load_vlm, load_swift_template

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def load_model_with_aux_head(args: argparse.Namespace):
    model, resolved_attention = load_vlm(
        args.model, attn_implementation=args.attn_implementation
    )
    if args.adapters:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("peft is not importable; install it with 'pip install peft'.") from exc
        model = PeftModel.from_pretrained(model, args.adapters)
        model = model.merge_and_unload() if getattr(args, "merge_lora", False) else model
    head = AuxHead.load(args.aux_head, device=model.device)
    logger.info(
        "loaded aux head from %s (hidden_size=%d, num_classes=%d)",
        args.aux_head,
        head.hidden_size,
        head.num_classes,
    )
    return model, head, resolved_attention


def run_head_inference(
    model: Any,
    head: AuxHead,
    template: Any,
    system_prompt: str,
    records: list[dict],
    batch_size: int,
    dump_decoder: bool = False,
    label_token_ids: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Return aux-head probabilities/logits plus optional decoder-argmax dumps.

    The decoder dump reuses the aux head's last-token hidden state and projects
    it through the language-model head, so both readouts come from exactly the
    same position. Returns ``(probabilities, head_logits, decoder_label_logits,
    decoder_probabilities, decoder_argmax_token_ids)``; the decoder dumps are
    None when ``dump_decoder`` is off.
    """

    def load_images(batch: list[dict]) -> list[Image.Image]:
        return [Image.open(r["images"][0]).convert("RGB") for r in batch]

    probabilities = np.zeros((len(records), head.num_classes), dtype=np.float32)
    head_logits_all = np.zeros((len(records), head.num_classes), dtype=np.float32)
    decoder_label_logits = (
        np.zeros((len(records), head.num_classes), dtype=np.float32) if dump_decoder else None
    )
    decoder_probabilities = (
        np.zeros((len(records), head.num_classes), dtype=np.float32) if dump_decoder else None
    )
    decoder_argmax = np.full(len(records), -1, dtype=np.int64) if dump_decoder else None
    lm_head = None
    label_token_id_tensor = None
    if dump_decoder:
        if not label_token_ids:
            raise ValueError("--dump-decoder-argmax requires the label token ids")
        lm_head = model.get_output_embeddings()
        if lm_head is None:
            raise RuntimeError("model has no output embedding head; cannot dump decoder argmax")
        label_token_id_tensor = torch.tensor(
            list(label_token_ids.values()), device=model.device, dtype=torch.long
        )
    batch_starts = list(range(0, len(records), batch_size))
    with ThreadPoolExecutor(max_workers=2) as image_pool:
        future = image_pool.submit(load_images, records[:batch_size])
        with tqdm(total=len(records), desc="aux-head eval", unit="img", leave=False) as progress:
            for batch_index, start in enumerate(batch_starts):
                batch = records[start : start + batch_size]
                images = future.result()
                if batch_index + 1 < len(batch_starts):
                    next_start = batch_starts[batch_index + 1]
                    future = image_pool.submit(load_images, records[next_start : next_start + batch_size])

                inputs = _encode_batch(model, template, system_prompt, batch, images)
                _, inputs = template.pre_forward_hook(model, None, inputs)
                kwargs = dict(use_cache=False, output_hidden_states=True, **keep_last_logits_kwargs(model))
                with torch.inference_mode():
                    try:
                        outputs = model(**inputs, **kwargs)
                    except TypeError:
                        outputs = model(**inputs, use_cache=False, output_hidden_states=True)
                    hidden = outputs.hidden_states[-1]
                    last_hidden = gather_last_hidden(hidden, inputs["attention_mask"])
                    logits = head(last_hidden).float()
                head_logits_all[start : start + len(batch)] = logits.cpu().numpy()
                probabilities[start : start + len(batch)] = torch.sigmoid(logits).cpu().numpy()
                if dump_decoder:
                    last_vocab_logits = lm_head(last_hidden).float()
                    decoder_argmax[start : start + len(batch)] = (
                        last_vocab_logits.argmax(dim=-1).cpu().numpy()
                    )
                    batch_decoder_label_logits = last_vocab_logits[:, label_token_id_tensor]
                    decoder_label_logits[start : start + len(batch)] = (
                        batch_decoder_label_logits.cpu().numpy()
                    )
                    decoder_probabilities[start : start + len(batch)] = torch.sigmoid(
                        batch_decoder_label_logits
                    ).cpu().numpy()
                progress.update(len(batch))
    return probabilities, head_logits_all, decoder_label_logits, decoder_probabilities, decoder_argmax


def summarize_distribution(
    y_true: np.ndarray, probabilities: np.ndarray, label_codes: list[str]
) -> dict[str, Any]:
    distribution: dict[str, Any] = {}
    for index, code in enumerate(label_codes):
        positive = y_true[:, index] == 1
        negative = y_true[:, index] == 0
        distribution[code] = {
            "mean_probability": float(probabilities[:, index].mean()),
            "fraction_ge_0.5": float((probabilities[:, index] >= 0.5).mean()),
            "positive_mean_probability": (
                float(probabilities[positive, index].mean()) if positive.any() else None
            ),
            "negative_mean_probability": (
                float(probabilities[negative, index].mean()) if negative.any() else None
            ),
        }
    return distribution


def compute_strict_argmax_metrics(
    y_true: np.ndarray,
    decoder_argmax: np.ndarray,
    label_token_ids: dict[str, int],
    primary_codes: list[str | None],
) -> dict[str, Any]:
    """Strict primary metrics from the decoder's full-vocabulary argmax.

    Mirrors the main-method definitions: designated-primary accuracy compares
    the raw argmax token with the record's protocol-designated primary code;
    any-positive accuracy credits an in-set argmax that hits any true label.
    """
    token_to_index = {token_id: index for index, token_id in enumerate(label_token_ids.values())}
    code_to_token = label_token_ids
    designated_hits = designated_total = any_hits = any_total = 0
    for row, raw_token_id in enumerate(decoder_argmax):
        token_id = int(raw_token_id)
        primary_code = primary_codes[row]
        if primary_code is not None:
            designated_total += 1
            designated_hits += int(token_id == code_to_token[primary_code])
        if y_true[row].sum() > 0:
            any_total += 1
            label_index = token_to_index.get(token_id)
            any_hits += int(label_index is not None and y_true[row, label_index] == 1.0)
    return {
        "strict_designated_primary_accuracy": (
            designated_hits / designated_total if designated_total else None
        ),
        "strict_any_positive_accuracy": any_hits / any_total if any_total else None,
        "num_designated_primary_samples": designated_total,
        "num_any_positive_samples": any_total,
    }


def evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.dataset)
    records = load_jsonl(args.test_data)
    records.sort(key=lambda r: Path(r["images"][0]).stat().st_size)
    validate_first_positive_records(
        config, records, source="evaluation records", allow_no_positive=True
    )
    y_true = build_label_matrix(records, config.num_classes)
    model, head, _ = load_model_with_aux_head(args)
    template, tokenizer = load_swift_template(args.model, model)
    label_token_ids = config.get_label_token_ids(tokenizer)

    torch.set_float32_matmul_precision("high")
    (
        probabilities,
        head_logits,
        decoder_label_logits,
        decoder_probabilities,
        decoder_argmax,
    ) = run_head_inference(
        model,
        head,
        template,
        config.render_prompt(),
        records,
        args.batch_size,
        dump_decoder=args.dump_decoder_argmax,
        label_token_ids=label_token_ids,
    )

    mean_ap, per_class_ap = compute_map(y_true, probabilities)
    mean_accuracy, per_class_accuracy = compute_accuracy(y_true, probabilities)
    mean_auc, per_class_auc = compute_auc(y_true, probabilities)
    ece, _ = compute_ece(y_true, probabilities)
    brier, _ = compute_brier(y_true, probabilities)
    nll, _ = compute_nll(y_true, probabilities)

    strict_metrics = {
        "strict_designated_primary_accuracy": None,
        "strict_any_positive_accuracy": None,
    }
    if decoder_argmax is not None:
        strict_metrics = compute_strict_argmax_metrics(
            y_true, decoder_argmax, label_token_ids, [r.get("primary_code") for r in records]
        )

    payload = {
        "dataset": config.name,
        "num_classes": config.num_classes,
        "metric_protocol_version": 3,
        "primary_label_protocol": "first_positive_config_order",
        "sigmoid_precision": "float32",
        "readout": "aux_head_last_token_hidden",
        "decoder_argmax_dumped": decoder_argmax is not None,
        "num_test_images": len(records),
        "num_zero_positive_samples": int((y_true.sum(axis=1) == 0).sum()),
        "mean_ap": mean_ap,
        "strict_designated_primary_accuracy": strict_metrics[
            "strict_designated_primary_accuracy"
        ],
        "strict_any_positive_accuracy": strict_metrics["strict_any_positive_accuracy"],
        "mean_accuracy": mean_accuracy,
        "mean_auc": mean_auc,
        "ece": ece,
        "brier": brier,
        "nll": nll,
        "threshold_sweep": compute_threshold_sweep(y_true, probabilities),
        "prediction_distribution": summarize_distribution(
            y_true, probabilities, config.label_codes
        ),
        "per_class_ap": {
            code: float(ap)
            for code, ap in zip(config.label_codes, per_class_ap, strict=True)
            if not np.isnan(ap)
        },
        "per_class_accuracy": {
            code: float(value)
            for code, value in zip(config.label_codes, per_class_accuracy, strict=True)
            if not np.isnan(value)
        },
        "per_class_auc": {
            code: float(value)
            for code, value in zip(config.label_codes, per_class_auc, strict=True)
            if not np.isnan(value)
        },
    }
    if decoder_argmax is not None:
        payload["num_designated_primary_samples"] = strict_metrics[
            "num_designated_primary_samples"
        ]
        payload["num_any_positive_samples"] = strict_metrics["num_any_positive_samples"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("results saved to %s", output_path)
    logger.info("mAP: %.4f  BalAcc@0.5: %.4f  AUC: %.4f  ECE: %.4f", mean_ap, mean_accuracy, mean_auc, ece)
    if decoder_argmax is not None:
        logger.info(
            "strict designated primary: %.4f  strict any-positive: %.4f",
            strict_metrics["strict_designated_primary_accuracy"],
            strict_metrics["strict_any_positive_accuracy"],
        )

    per_sample_output = args.per_sample_output or str(output_path.with_suffix(".per_sample.jsonl"))
    per_sample_path = Path(per_sample_output)
    per_sample_path.parent.mkdir(parents=True, exist_ok=True)
    token_to_code = {token_id: code for code, token_id in label_token_ids.items()}
    with per_sample_path.open("w", encoding="utf-8") as stream:
        for row, record in enumerate(records):
            row_payload = {
                "image": record["images"][0],
                "y_true": y_true[row].astype(float).tolist(),
                "head_logits": head_logits[row].astype(float).tolist(),
                "probabilities": probabilities[row].astype(float).tolist(),
            }
            if decoder_argmax is not None:
                argmax_token_id = int(decoder_argmax[row])
                label_index = {
                    token_id: index for index, token_id in enumerate(label_token_ids.values())
                }.get(argmax_token_id)
                primary_token_id = label_token_ids.get(record.get("primary_code"))
                row_payload.update(
                    {
                        "decoder_label_logits": decoder_label_logits[row].astype(float).tolist(),
                        "decoder_probabilities": decoder_probabilities[row]
                        .astype(float)
                        .tolist(),
                        "argmax_token_id": argmax_token_id,
                        "argmax_in_label_set": label_index is not None,
                        "argmax_code": token_to_code.get(argmax_token_id),
                        "designated_primary_code": record.get("primary_code"),
                        "designated_primary_correct": (
                            None
                            if primary_token_id is None
                            else bool(argmax_token_id == primary_token_id)
                        ),
                        "any_positive_correct": (
                            None
                            if y_true[row].sum() == 0
                            else bool(label_index is not None and y_true[row, label_index] == 1.0)
                        ),
                    }
                )
            stream.write(json.dumps(row_payload, ensure_ascii=False) + "\n")
    logger.info("per-sample results saved to %s", per_sample_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapters", default="", help="LoRA adapter directory (peft format)")
    parser.add_argument("--aux-head", required=True, help="aux_head.pt path")
    parser.add_argument("--dataset", required=True, choices=["voc", "chestxray14", "celeba"])
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--attn-implementation", default="flash_attention_2"
    )
    parser.add_argument("--per-sample-output", default=None)
    parser.add_argument(
        "--dump-decoder-argmax",
        action="store_true",
        help="also project the last-token hidden state through the LM head "
        "to report the decoder's strict argmax metrics and per-sample "
        "decoder fields",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    evaluate(parse_args(argv))


if __name__ == "__main__":
    main()
