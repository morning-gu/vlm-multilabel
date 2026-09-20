"""Offline post-hoc calibration from saved per-sample label-token logits.

Reads per-sample JSONL files written by `vlm-multilabel evaluate` (fields
``y_true`` and ``label_logits``), fits temperature scaling by minimizing NLL
on the evaluated split, and reports raw vs temperature-scaled metrics plus
the optimal-threshold balanced accuracy.

Caveat: the temperature is fitted on the same split it is reported on, so
treat the scaled numbers as a diagnostic upper bound, not held-out results.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from vlm_multilabel.metrics import (
    compute_brier,
    compute_ece,
    compute_map,
    compute_nll,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def load_per_sample(path: str) -> tuple[np.ndarray, np.ndarray]:
    y_true_rows: list[list[float]] = []
    logits_rows: list[list[float]] = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            logits_field = (
                "label_logits" if "label_logits" in payload else "head_logits"
            )
            if "y_true" not in payload or logits_field not in payload:
                raise ValueError(
                    f"{path}:{line_number} lacks y_true/{{label,head}}_logits"
                )
            y_true_rows.append(payload["y_true"])
            logits_rows.append(payload[logits_field])
    y_true = np.asarray(y_true_rows, dtype=np.float32)
    logits = np.asarray(logits_rows, dtype=np.float32)
    if y_true.shape != logits.shape:
        raise ValueError(f"{path}: y_true {y_true.shape} != label_logits {logits.shape}")
    return y_true, logits


def fit_temperature(y_true: np.ndarray, logits: np.ndarray) -> float:
    """Fit T >= 0 by minimizing split NLL over sigmoid(logits / T)."""
    def scaled_nll(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        probabilities = 1.0 / (1.0 + np.exp(-logits / temperature))
        value, _ = compute_nll(y_true, probabilities)
        return float(value)

    result = minimize_scalar(scaled_nll, bounds=(-4.0, 4.0), method="bounded")
    return float(np.exp(result.x))


def balanced_accuracy_at_threshold(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> float:
    predictions = (probabilities >= threshold).astype(np.float32)
    per_class: list[float] = []
    for index in range(y_true.shape[1]):
        target = y_true[:, index]
        positive = target == 1
        negative = target == 0
        if not positive.any() or not negative.any():
            continue
        tpr = float(predictions[positive, index].mean())
        tnr = float(1.0 - predictions[negative, index].mean())
        per_class.append((tpr + tnr) / 2.0)
    return float(np.mean(per_class)) if per_class else float("nan")


def best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    scores = np.array(
        [balanced_accuracy_at_threshold(y_true, probabilities, t) for t in grid]
    )
    best_index = int(np.nanargmax(scores))
    return {
        "threshold": float(grid[best_index]),
        "balanced_accuracy": float(scores[best_index]),
        "balanced_accuracy_at_0.5": balanced_accuracy_at_threshold(y_true, probabilities, 0.5),
    }


def evaluate_file(path: str) -> dict[str, object]:
    y_true, logits = load_per_sample(path)
    raw_probabilities = 1.0 / (1.0 + np.exp(-logits))
    temperature = fit_temperature(y_true, logits)
    scaled_probabilities = 1.0 / (1.0 + np.exp(-logits / temperature))

    def metric_block(probabilities: np.ndarray) -> dict[str, object]:
        ece, _ = compute_ece(y_true, probabilities)
        brier, _ = compute_brier(y_true, probabilities)
        nll, _ = compute_nll(y_true, probabilities)
        mean_ap, _ = compute_map(y_true, probabilities)
        return {
            "mean_ap": float(mean_ap),
            "ece": float(ece),
            "brier": float(brier),
            "nll": float(nll),
            **best_threshold(y_true, probabilities),
        }

    return {
        "per_sample_file": path,
        "num_samples": int(y_true.shape[0]),
        "num_classes": int(y_true.shape[1]),
        "temperature": temperature,
        "raw": metric_block(raw_probabilities),
        "temperature_scaled": metric_block(scaled_probabilities),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-sample", nargs="+", required=True, help="per-sample JSONL files (globs allowed)")
    parser.add_argument("--output", required=True, help="output JSON path")
    args = parser.parse_args(argv)

    paths: list[str] = []
    for pattern in args.per_sample:
        matched = sorted(str(p) for p in Path(".").glob(pattern))
        paths.extend(matched if matched else [pattern])

    results = []
    for path in paths:
        logger.info("processing %s", path)
        results.append(evaluate_file(path))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("post-hoc calibration results saved to %s", output_path)


if __name__ == "__main__":
    main()
