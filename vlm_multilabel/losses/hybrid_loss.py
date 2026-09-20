"""Hybrid loss: next-token cross-entropy plus MP correction and BCE."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from vlm_multilabel.losses.label_positions import find_label_positions

MP_MARGIN = 2.0


def label_position_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_vector: torch.Tensor,
    label_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute BCE over label-token logits, averaged over valid samples."""
    device = logits.device
    token_ids = label_token_ids.to(device)
    batch_indices = torch.arange(logits.shape[0], device=device)
    positions, valid = find_label_positions(labels, token_ids)
    label_logits = logits[batch_indices, positions][:, token_ids]
    targets = label_vector.to(device).float()
    per_sample_bce = F.binary_cross_entropy_with_logits(
        label_logits, targets, reduction="none"
    ).mean(dim=1)
    valid_float = valid.float()
    return (per_sample_bce * valid_float).sum() / valid_float.sum().clamp(min=1.0)


def label_position_mp_correction_setrank(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_vector: torch.Tensor,
    label_token_ids: torch.Tensor,
    cross_entropy_denominator: float,
) -> torch.Tensor:
    """Compute the set-level multi-positive margin-ranking correction.

    Only samples with at least two positive labels and at least one negative
    label contribute, preventing degenerate one-class sets.
    """
    device = logits.device
    token_ids = label_token_ids.to(device)
    num_classes = token_ids.shape[0]
    batch_indices = torch.arange(logits.shape[0], device=device)
    positions, valid = find_label_positions(labels, token_ids)
    label_logits = logits[batch_indices, positions][:, token_ids]

    positive_mask = label_vector.to(device) == 1
    masked_valid = label_logits.masked_fill(~positive_mask, float("-inf"))
    masked_invalid = label_logits.masked_fill(positive_mask, float("-inf"))
    log_sum_valid = torch.logsumexp(masked_valid, dim=1)
    log_sum_invalid = torch.logsumexp(masked_invalid, dim=1)
    ranking_loss = F.softplus(MP_MARGIN + log_sum_invalid - log_sum_valid)

    positive_counts = positive_mask.sum(dim=1)
    is_multi_label = (positive_counts > 1) & (positive_counts < num_classes) & valid
    correction = torch.where(is_multi_label, ranking_loss, torch.zeros_like(ranking_loss))
    return correction.sum() / float(cross_entropy_denominator)


# Backward-compatible alias retained while callers migrate.
