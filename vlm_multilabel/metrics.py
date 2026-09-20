"""Multi-label metrics shared by training, evaluation, and theory checks."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_map(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, list[float]]:
    """Return mean average precision and per-class AP values.

    Classes without positive labels are skipped, matching multi-label benchmark
    convention.
    """
    _validate_metric_inputs(y_true, y_pred)
    per_class = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    for index in range(y_true.shape[1]):
        if y_true[:, index].sum() > 0:
            per_class[index] = float(
                average_precision_score(y_true[:, index], y_pred[:, index])
            )
    valid = per_class[~np.isnan(per_class)]
    return (float(valid.mean()) if valid.size else 0.0), per_class.tolist()


def compute_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> tuple[float, np.ndarray]:
    """Return mean balanced accuracy and a per-class array (NaN if skipped)."""
    _validate_metric_inputs(y_true, y_pred)
    y_hat = (y_pred >= threshold).astype(np.float32)
    per_class = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    for index in range(y_true.shape[1]):
        positives = y_true[:, index] == 1
        negatives = ~positives
        if not positives.any() or not negatives.any():
            continue
        tpr = float(y_hat[positives, index].mean())
        tnr = float((y_hat[negatives, index] == 0).mean())
        per_class[index] = (tpr + tnr) / 2.0
    return float(np.nanmean(per_class)), per_class


def compute_threshold_sweep(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: tuple[float, ...] = (
        0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60,
        0.70, 0.80, 0.90, 0.95, 0.99,
    ),
) -> list[dict[str, float]]:
    """Return macro balanced accuracy over a fixed threshold grid."""
    _validate_metric_inputs(y_true, y_pred)
    return [
        {
            "threshold": float(threshold),
            "mean_balanced_accuracy": compute_accuracy(y_true, y_pred, threshold)[0],
        }
        for threshold in thresholds
    ]


def compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
    """Return mean AUROC and a per-class array (NaN if skipped)."""
    _validate_metric_inputs(y_true, y_pred)
    per_class = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    for index in range(y_true.shape[1]):
        targets = y_true[:, index]
        if targets.min() == targets.max():
            continue
        per_class[index] = float(roc_auc_score(targets, y_pred[:, index]))
    return float(np.nanmean(per_class)), per_class


def _validate_metric_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"metric shape mismatch: {y_true.shape=} {y_pred.shape=}")
    if y_true.ndim != 2:
        raise ValueError("metrics expect [num_samples, num_classes] arrays")


def compute_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_bins: int = 15,
) -> tuple[float, list[float]]:
    """Return macro per-label ECE and per-label ECE using equal-mass bins.

    For multilabel outputs, calibration is measured independently for every
    label and then macro-averaged over classes with both positive and negative
    examples.
    """
    _validate_metric_inputs(y_true, y_pred)
    per_label = []
    for index in range(y_true.shape[1]):
        targets = y_true[:, index]
        probabilities = y_pred[:, index]
        if targets.min() == targets.max():
            continue
        edges = np.quantile(probabilities, np.linspace(0, 1, num_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0 + np.finfo(np.float64).eps
        ece = 0.0
        for left, right in zip(edges[:-1], edges[1:], strict=True):
            mask = (probabilities >= left) & (probabilities < right)
            if not mask.any():
                continue
            ece += mask.mean() * abs(targets[mask].mean() - probabilities[mask].mean())
        per_label.append(float(ece))
    return (float(np.mean(per_label)) if per_label else 0.0), per_label


def compute_brier(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, list[float]]:
    """Return macro per-label Brier score and per-label Brier scores."""
    _validate_metric_inputs(y_true, y_pred)
    per_label = [
        float(np.mean((y_pred[:, index] - y_true[:, index]) ** 2))
        for index in range(y_true.shape[1])
        if y_true[:, index].min() != y_true[:, index].max()
    ]
    return (float(np.mean(per_label)) if per_label else 0.0), per_label


def compute_nll(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-7,
) -> tuple[float, list[float]]:
    """Return macro per-label binary NLL and per-label NLL values."""
    _validate_metric_inputs(y_true, y_pred)
    probabilities = np.clip(y_pred.astype(np.float64), eps, 1.0 - eps)
    per_label = []
    for index in range(y_true.shape[1]):
        targets = y_true[:, index]
        if targets.min() == targets.max():
            continue
        values = -(
            targets * np.log(probabilities[:, index])
            + (1.0 - targets) * np.log(1.0 - probabilities[:, index])
        )
        per_label.append(float(values.mean()))
    return (float(np.mean(per_label)) if per_label else 0.0), per_label
