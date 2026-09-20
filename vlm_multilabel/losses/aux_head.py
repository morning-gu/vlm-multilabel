"""Independent classification head baseline (aux-head control).

The control keeps the same backbone, the same input encoding, and the same
readout position as the main method, but applies multi-label supervision
through a separate linear head on the last-token hidden state instead of the
decoder label-token logits. No next-token CE, no MP, no BCE on the vocabulary.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def gather_last_hidden(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Gather the hidden state at each sample's last real token.

    Works for both left and right padding by counting trailing zeros in the
    attention mask: ``last_index = L - 1 - (right pad count)``.
    """
    sequence_length = attention_mask.shape[1]
    positions = torch.arange(sequence_length, device=attention_mask.device)
    flipped_valid = attention_mask.flip(-1) > 0
    first_real_in_flipped = torch.where(
        flipped_valid,
        positions.unsqueeze(0),
        torch.full_like(positions, sequence_length),
    ).min(dim=1).values
    last_indices = (sequence_length - 1 - first_real_in_flipped).clamp(min=0)
    batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_indices, last_indices]


def aux_head_bce_loss(
    head_logits: torch.Tensor, label_vector: torch.Tensor
) -> torch.Tensor:
    """Mean BCE over all classes and samples; labels are the full label_vector."""
    targets = label_vector.to(head_logits.device).float()
    return F.binary_cross_entropy_with_logits(head_logits, targets, reduction="mean")


class AuxHead(nn.Module):
    """Linear readout from the backbone's last-token hidden state."""

    def __init__(self, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden.float())

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "hidden_size": self.hidden_size,
                "num_classes": self.num_classes,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: torch.device | str = "cpu") -> "AuxHead":
        payload = torch.load(path, map_location=device, weights_only=True)
        head = cls(payload["hidden_size"], payload["num_classes"])
        head.load_state_dict(payload["state_dict"])
        head.to(device)
        return head
