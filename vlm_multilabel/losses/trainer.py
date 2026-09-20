"""Hybrid-loss Trainer factory and mAP logging."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from vlm_multilabel.losses.label_positions import extract_label_probabilities
from vlm_multilabel.losses.hybrid_loss import label_position_bce, label_position_mp_correction_setrank
from vlm_multilabel.metrics import compute_map

logger = logging.getLogger(__name__)


class _MapAccumulator:
    """Accumulate multi-label predictions and calculate mean AP."""

    def __init__(self) -> None:
        self.targets: list[np.ndarray] = []
        self.probabilities: list[np.ndarray] = []

    def extend(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        self.targets.extend(y_true)
        self.probabilities.extend(y_pred)

    def mean_map(self) -> float | None:
        if not self.probabilities:
            return None
        y_true = np.asarray(self.targets)
        y_pred = np.asarray(self.probabilities)
        mean_map, _ = compute_map(y_true, y_pred)
        return mean_map

    def clear(self) -> None:
        self.targets.clear()
        self.probabilities.clear()


def make_trainer(label_ids: dict[str, int], bce_weight: float = 1.0, mp_weight: float = 1.0):
    """Create a Swift trainer that adds BCE/MP losses and in-loop mAP logging."""
    from swift.trainers import Seq2SeqTrainer

    label_token_ids = (
        torch.tensor(list(label_ids.values()), dtype=torch.long) if label_ids else None
    )

    class HybridLossTrainer(Seq2SeqTrainer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._train_map = _MapAccumulator()
            self._eval_map = _MapAccumulator()

        def _accumulate_predictions(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            label_vector: torch.Tensor,
            is_eval: bool,
        ) -> None:
            probabilities, valid = extract_label_probabilities(
                logits, labels, label_token_ids
            )
            valid_rows = valid.cpu().numpy()
            probabilities = probabilities.detach().float().cpu().numpy()
            targets = label_vector.detach().float().cpu().numpy()
            accumulator = self._eval_map if is_eval else self._train_map
            accumulator.extend(targets[valid_rows], probabilities[valid_rows])

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            label_vector = inputs.pop("label_vector", None)
            labels = inputs.get("labels")
            loss, outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
            logits = getattr(outputs, "logits", None)
            if label_token_ids is None or label_vector is None or labels is None or logits is None:
                return (loss, outputs) if return_outputs else loss

            label_vector = label_vector.to(logits.device)
            if num_items_in_batch is not None:
                denominator = float(num_items_in_batch)
            else:
                denominator = float((labels != -100).sum().item()) or 1.0
            if mp_weight:
                loss = loss + mp_weight * label_position_mp_correction_setrank(
                    logits, labels, label_vector, label_token_ids, denominator
                )
            loss = loss + bce_weight * label_position_bce(
                logits, labels, label_vector, label_token_ids
            )
            self._accumulate_predictions(
                logits, labels, label_vector, is_eval=not self.model.training
            )
            return (loss, outputs) if return_outputs else loss

        def evaluate(self, *args: Any, **kwargs: Any) -> Any:
            self._eval_map.clear()
            return super().evaluate(*args, **kwargs)

        def log(self, logs: dict[str, Any], *args: Any, **kwargs: Any) -> None:
            eval_map = self._eval_map.mean_map()
            if eval_map is not None and "eval_loss" in logs:
                logs["eval_map"] = eval_map
            train_map = self._train_map.mean_map()
            if (
                train_map is not None
                and "loss" in logs
                and "eval_loss" not in logs
                and "train_loss" not in logs
            ):
                logs["train_map"] = train_map
                self._train_map.clear()
            super().log(logs, *args, **kwargs)

    return HybridLossTrainer
