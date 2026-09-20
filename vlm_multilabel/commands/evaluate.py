"""Evaluate a trained VLM on multi-label classification.

The script performs one non-autoregressive forward pass per batch, extracts
label-token logits at the last content position, and evaluates mAP, balanced
accuracy, AUROC, two strict primary metrics, threshold sweep, and calibration.
It also writes per-sample logits/probabilities to JSONL.
"""
from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from vlm_multilabel.constants.base import DatasetConfig, load_config
from vlm_multilabel.data.conversion import validate_first_positive_records
from vlm_multilabel.metrics import (
    compute_accuracy,
    compute_auc,
    compute_brier,
    compute_ece,
    compute_map,
    compute_nll,
    compute_threshold_sweep,
)
from vlm_multilabel.runtime.model import (
    build_label_matrix,
    keep_last_logits_kwargs,
    load_jsonl,
    load_swift_template,
    load_vlm,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationData:
    records: list[dict]
    y_true: np.ndarray
    label_token_ids: dict[str, int]
    primary_token_ids: np.ndarray
    primary_codes: list[str | None]
    system_prompt: str

    @property
    def token_ids(self) -> list[int]:
        return list(self.label_token_ids.values())

    @property
    def token_to_code(self) -> dict[int, str]:
        return {token_id: code for code, token_id in self.label_token_ids.items()}


@dataclass(frozen=True)
class Predictions:
    probabilities: np.ndarray
    argmax_token_ids: np.ndarray
    label_logits: np.ndarray


@dataclass(frozen=True)
class EvaluationResults:
    mean_ap: float
    per_class_ap: list[float]
    mean_accuracy: float
    per_class_accuracy: np.ndarray
    mean_auc: float
    per_class_auc: np.ndarray
    designated_primary_accuracy: float
    any_positive_accuracy: float
    num_designated_primary_samples: int
    num_any_positive_samples: int
    num_zero_positive_samples: int
    ece: float
    brier: float
    nll: float


def prepare_evaluation_data(
    config: DatasetConfig,
    records: list[dict],
    tokenizer: Any,
) -> EvaluationData:
    """Load, order, tokenize, and materialize labels for evaluation."""
    records.sort(key=lambda record: Path(record["images"][0]).stat().st_size)
    validate_first_positive_records(
        config, records, source="evaluation records", allow_no_positive=True
    )
    label_token_ids = config.get_label_token_ids(tokenizer)
    primary_codes = [record.get("primary_code") for record in records]
    missing_primary_tokens = [
        code for code in primary_codes
        if code is not None and code not in label_token_ids
    ]
    if missing_primary_tokens:
        raise ValueError(
            "Primary codes are not single-token label IDs: "
            f"{sorted(set(missing_primary_tokens))}"
        )
    primary_token_ids = np.asarray(
        [
            -1 if code is None else label_token_ids[code]
            for code in primary_codes
        ],
        dtype=np.int64,
    )
    return EvaluationData(
        records=records,
        y_true=build_label_matrix(records, config.num_classes),
        label_token_ids=label_token_ids,
        primary_token_ids=primary_token_ids,
        primary_codes=primary_codes,
        system_prompt=config.render_prompt(),
    )


def _encode_batch(
    model: Any,
    template: Any,
    system_prompt: str,
    records: list[dict],
    images: list[Image.Image],
) -> dict[str, Any]:
    from swift import InferRequest

    requests = [
        InferRequest(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": record["messages"][1]["content"]},
            ],
            images=[image],
        )
        for record, image in zip(records, images)
    ]
    encoded = [template.encode(request, return_template_inputs=True) for request in requests]
    for item in encoded:
        item.pop("template_inputs", None)
    inputs = template.data_collator(encoded)
    return {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def run_inference(
    model: Any,
    template: Any,
    evaluation_data: EvaluationData,
    batch_size: int,
) -> Predictions:
    """Run single-forward-pass inference and extract label-token probabilities."""
    token_ids = evaluation_data.token_ids
    probabilities = np.zeros((len(evaluation_data.records), len(token_ids)), dtype=np.float32)
    label_logits = np.zeros((len(evaluation_data.records), len(token_ids)), dtype=np.float32)
    argmax_token_ids = np.full(len(evaluation_data.records), -1, dtype=np.int64)
    token_id_tensor = torch.tensor(token_ids, device=model.device, dtype=torch.long)

    def load_images(records: list[dict]) -> list[Image.Image]:
        return [Image.open(record["images"][0]).convert("RGB") for record in records]

    with ThreadPoolExecutor(max_workers=2) as image_pool:
        batch_starts = list(range(0, len(evaluation_data.records), batch_size))
        future = image_pool.submit(load_images, evaluation_data.records[:batch_size])
        with tqdm(
            total=len(evaluation_data.records),
            desc="eval",
            unit="img",
            leave=False,
        ) as progress:
            for batch_index, start in enumerate(batch_starts):
                records = evaluation_data.records[start : start + batch_size]
                images = future.result()
                if batch_index + 1 < len(batch_starts):
                    next_start = batch_starts[batch_index + 1]
                    future = image_pool.submit(
                        load_images, evaluation_data.records[next_start : next_start + batch_size]
                    )

                inputs = _encode_batch(
                    model, template, evaluation_data.system_prompt, records, images
                )
                _, inputs = template.pre_forward_hook(model, None, inputs)
                with torch.inference_mode():
                    forward_kwargs = dict(
                        use_cache=False, **keep_last_logits_kwargs(model)
                    )
                    try:
                        outputs = model(**inputs, **forward_kwargs)
                    except TypeError:
                        outputs = model(**inputs, use_cache=False)
                    last_logits = outputs.logits[:, -1, :]
                    predicted_tokens = last_logits.argmax(dim=-1).tolist()
                    batch_label_logits = last_logits[:, token_id_tensor]
                    batch_label_logits_cpu = batch_label_logits.float().cpu().numpy()
                    batch_probabilities = torch.sigmoid(batch_label_logits.float()).cpu().numpy()

                for row, (token_id, row_logits, row_probabilities) in enumerate(
                    zip(predicted_tokens, batch_label_logits_cpu, batch_probabilities, strict=True)
                ):
                    argmax_token_ids[start + row] = token_id
                    label_logits[start + row] = row_logits
                    probabilities[start + row] = row_probabilities

                progress.update(len(records))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return Predictions(probabilities, argmax_token_ids, label_logits)


def calculate_results(
    evaluation_data: EvaluationData,
    predictions: Predictions,
) -> EvaluationResults:
    """Compute all evaluation metrics from collected predictions."""
    mean_ap, per_class_ap = compute_map(evaluation_data.y_true, predictions.probabilities)
    mean_accuracy, per_class_accuracy = compute_accuracy(
        evaluation_data.y_true, predictions.probabilities
    )
    mean_auc, per_class_auc = compute_auc(evaluation_data.y_true, predictions.probabilities)

    token_to_index = {token_id: index for index, token_id in enumerate(evaluation_data.token_ids)}
    designated_primary_hits = 0
    designated_primary_total = 0
    any_positive_hits = 0
    any_positive_total = 0
    for row, raw_token_id in enumerate(predictions.argmax_token_ids):
        token_id = int(raw_token_id)
        primary_token_id = int(evaluation_data.primary_token_ids[row])
        if primary_token_id >= 0:
            designated_primary_total += 1
            designated_primary_hits += int(token_id == primary_token_id)
        if evaluation_data.y_true[row].sum() > 0:
            any_positive_total += 1
            label_index = token_to_index.get(token_id)
            any_positive_hits += int(
                label_index is not None
                and evaluation_data.y_true[row, label_index] == 1.0
            )
    designated_primary_accuracy = (
        designated_primary_hits / designated_primary_total
        if designated_primary_total
        else 0.0
    )
    any_positive_accuracy = (
        any_positive_hits / any_positive_total if any_positive_total else 0.0
    )
    ece, _ = compute_ece(evaluation_data.y_true, predictions.probabilities)
    brier, _ = compute_brier(evaluation_data.y_true, predictions.probabilities)
    nll, _ = compute_nll(evaluation_data.y_true, predictions.probabilities)
    return EvaluationResults(
        mean_ap=mean_ap,
        per_class_ap=per_class_ap,
        mean_accuracy=mean_accuracy,
        per_class_accuracy=per_class_accuracy,
        mean_auc=mean_auc,
        per_class_auc=per_class_auc,
        designated_primary_accuracy=designated_primary_accuracy,
        any_positive_accuracy=any_positive_accuracy,
        num_designated_primary_samples=designated_primary_total,
        num_any_positive_samples=any_positive_total,
        num_zero_positive_samples=int((evaluation_data.y_true.sum(axis=1) == 0).sum()),
        ece=ece,
        brier=brier,
        nll=nll,
    )


def print_report(
    config: DatasetConfig,
    evaluation_data: EvaluationData,
    results: EvaluationResults,
) -> None:
    """Log a human-readable evaluation report."""
    logger.info("strict designated primary accuracy: %.4f", results.designated_primary_accuracy)
    logger.info("strict any-positive accuracy: %.4f", results.any_positive_accuracy)
    logger.info("mAP: %.4f", results.mean_ap)
    logger.info("mean balanced accuracy: %.4f", results.mean_accuracy)
    logger.info("mean AUROC: %.4f", results.mean_auc)
    logger.info("ECE (15 equal-mass bins): %.4f", results.ece)
    logger.info("Brier score: %.4f", results.brier)
    logger.info("binary NLL: %.4f", results.nll)

    token_to_code = evaluation_data.token_to_code
    for token_id, ap in zip(evaluation_data.token_ids, results.per_class_ap, strict=True):
        if np.isnan(ap):
            continue
        code = token_to_code[token_id]
        logger.info("AP %s (%s): %.4f", code, config.code_to_name.get(code, code), ap)
    for index, token_id in enumerate(evaluation_data.token_ids):
        accuracy = results.per_class_accuracy[index]
        if np.isnan(accuracy):
            continue
        code = token_to_code[token_id]
        logger.info("accuracy %s (%s): %.4f", code, config.code_to_name.get(code, code), accuracy)
    for index, token_id in enumerate(evaluation_data.token_ids):
        auc = results.per_class_auc[index]
        if np.isnan(auc):
            continue
        code = token_to_code[token_id]
        logger.info("AUROC %s (%s): %.4f", code, config.code_to_name.get(code, code), auc)


def summarize_prediction_distribution(
    evaluation_data: EvaluationData,
    predictions: Predictions,
) -> dict[str, Any]:
    """Summarize per-label probability mass for saturation diagnostics."""
    token_to_code = evaluation_data.token_to_code
    distribution: dict[str, Any] = {}
    for index, token_id in enumerate(evaluation_data.token_ids):
        targets = evaluation_data.y_true[:, index]
        probabilities = predictions.probabilities[:, index]
        positive = targets == 1
        negative = targets == 0
        if positive.any() and negative.any():
            positive_mean = float(probabilities[positive].mean())
            negative_mean = float(probabilities[negative].mean())
        else:
            positive_mean = None
            negative_mean = None
        distribution[token_to_code[token_id]] = {
            "mean_probability": float(probabilities.mean()),
            "fraction_ge_0.5": float((probabilities >= 0.5).mean()),
            "positive_mean_probability": positive_mean,
            "negative_mean_probability": negative_mean,
        }
    return distribution


def write_per_sample_results(
    config: DatasetConfig,
    evaluation_data: EvaluationData,
    predictions: Predictions,
    output: str,
) -> None:
    """Write one JSONL row per test image for offline threshold/distribution analysis."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_to_code = evaluation_data.token_to_code
    with output_path.open("w", encoding="utf-8") as stream:
        for row in range(len(evaluation_data.records)):
            argmax_token_id = int(predictions.argmax_token_ids[row])
            designated_primary_token_id = int(evaluation_data.primary_token_ids[row])
            designated_primary_code = evaluation_data.primary_codes[row]
            has_positive_labels = bool(evaluation_data.y_true[row].sum() > 0)
            token_to_index = {
                token_id: index for index, token_id in enumerate(evaluation_data.token_ids)
            }
            label_index = token_to_index.get(argmax_token_id)
            in_label_set = label_index is not None
            payload = {
                "image": evaluation_data.records[row]["images"][0],
                "y_true": evaluation_data.y_true[row].astype(float).tolist(),
                "label_logits": predictions.label_logits[row].astype(float).tolist(),
                "probabilities": predictions.probabilities[row].astype(float).tolist(),
                "argmax_token_id": argmax_token_id,
                "argmax_in_label_set": in_label_set,
                "argmax_code": token_to_code.get(argmax_token_id),
                "designated_primary_code": designated_primary_code,
                "designated_primary_correct": (
                    None
                    if designated_primary_token_id < 0
                    else bool(argmax_token_id == designated_primary_token_id)
                ),
                "any_positive_correct": (
                    None
                    if not has_positive_labels
                    else bool(
                        label_index is not None
                        and evaluation_data.y_true[row, label_index] == 1.0
                    )
                ),
            }
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    logger.info("per-sample results saved to %s", output_path)


def result_payload(
    config: DatasetConfig,
    evaluation_data: EvaluationData,
    predictions: Predictions,
    results: EvaluationResults,
) -> dict[str, Any]:
    """Build the JSON-serializable evaluation result object."""
    token_to_code = evaluation_data.token_to_code
    num_records = len(evaluation_data.records)
    return {
        "dataset": config.name,
        "num_classes": config.num_classes,
        "metric_protocol_version": 3,
        "primary_label_protocol": "first_positive_config_order",
        "sigmoid_precision": "float32",
        "num_test_images": num_records,
        "num_zero_positive_samples": results.num_zero_positive_samples,
        "num_designated_primary_samples": results.num_designated_primary_samples,
        "num_any_positive_samples": results.num_any_positive_samples,
        "mean_ap": results.mean_ap,
        "strict_designated_primary_accuracy": results.designated_primary_accuracy,
        "strict_any_positive_accuracy": results.any_positive_accuracy,
        "mean_accuracy": results.mean_accuracy,
        "mean_auc": results.mean_auc,
        "ece": results.ece,
        "brier": results.brier,
        "nll": results.nll,
        "threshold_sweep": compute_threshold_sweep(
            evaluation_data.y_true, predictions.probabilities
        ),
        "prediction_distribution": summarize_prediction_distribution(
            evaluation_data, predictions
        ),
        "per_class_ap": {
            token_to_code[token_id]: float(ap)
            for token_id, ap in zip(evaluation_data.token_ids, results.per_class_ap, strict=True)
            if not np.isnan(ap)
        },
        "per_class_accuracy": {
            token_to_code[evaluation_data.token_ids[index]]: float(value)
            for index, value in enumerate(results.per_class_accuracy)
            if not np.isnan(value)
        },
        "per_class_auc": {
            token_to_code[evaluation_data.token_ids[index]]: float(value)
            for index, value in enumerate(results.per_class_auc)
            if not np.isnan(value)
        },
    }


def save_results(payload: dict[str, Any], output: str) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("results saved to %s", output_path)


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate a model checkpoint on a JSONL test split."""
    config = load_config(args.dataset)
    records = load_jsonl(args.test_data)
    model, resolved_attention = load_vlm(
        args.model, adapters=args.adapters, attn_implementation=args.attn_implementation
    )
    template, tokenizer = load_swift_template(args.model, model)
    evaluation_data = prepare_evaluation_data(config, records, tokenizer)
    logger.info(
        "dataset=%s records=%d classes=%d single-token-labels=%d attention=%s",
        config.name,
        len(records),
        config.num_classes,
        len(evaluation_data.label_token_ids),
        resolved_attention or "framework-default",
    )

    torch.set_float32_matmul_precision("high")
    predictions = run_inference(model, template, evaluation_data, args.batch_size)
    results = calculate_results(evaluation_data, predictions)
    print_report(config, evaluation_data, results)
    per_sample_output = args.per_sample_output or str(
        Path(args.output).with_suffix(".per_sample.jsonl")
    )
    save_results(result_payload(config, evaluation_data, predictions, results), args.output)
    write_per_sample_results(
        config, evaluation_data, predictions, per_sample_output
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="base model id/path")
    parser.add_argument("--adapters", default="", help="LoRA adapter directory")
    parser.add_argument(
        "--dataset", required=True, choices=["voc", "chestxray14", "celeba"]
    )
    parser.add_argument("--test-data", required=True, help="test JSONL path")
    parser.add_argument("--output", default="results/eval.json", help="output JSON path")
    parser.add_argument("--batch-size", type=int, default=16, help="images per forward pass")
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help="flash_attention_2 | sdpa | eager (empty=framework default)",
    )
    parser.add_argument(
        "--per-sample-output",
        default=None,
        help="per-sample JSONL path; defaults to <output>.per_sample.jsonl",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    evaluate(parse_args(argv))


if __name__ == "__main__":
    main()
