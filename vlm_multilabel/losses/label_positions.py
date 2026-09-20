"""Vectorized label-token position search and probability extraction."""
from __future__ import annotations

import torch


def find_label_positions(labels: torch.Tensor, label_token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the first label-token logit position for every sample.

    Args:
        labels: [batch, sequence] target ids, with -100 for non-response positions.
        label_token_ids: [num_classes] token ids.

    Returns:
        A ``(logit_positions, valid)`` tuple. Invalid samples are clamped to
        position zero and have a False validity mask.
    """
    response_mask = labels != -100
    label_mask = (labels.unsqueeze(-1) == label_token_ids).any(dim=-1)
    candidates = response_mask & label_mask
    has_label = candidates.any(dim=1)
    first_label_position = candidates.float().argmax(dim=1)
    logit_positions = (first_label_position - 1).clamp(min=0)
    valid = has_label & (first_label_position > 0)
    return logit_positions, valid


def extract_label_probabilities(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract float32 sigmoid probabilities for each label token."""
    device = logits.device
    token_ids = label_token_ids.to(device)
    batch_indices = torch.arange(logits.shape[0], device=device)
    positions, valid = find_label_positions(labels, token_ids)
    label_logits = logits[batch_indices, positions][:, token_ids]
    return torch.sigmoid(label_logits.float()), valid


# Backward-compatible names used by older research scripts.
