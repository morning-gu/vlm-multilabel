"""Patch ms-swift templates so per-sample label vectors survive collation."""
from __future__ import annotations

from typing import Any

import torch


def patch_template_for_label_vector(template: Any) -> None:
    """Thread `label_vector` through Swift encoding and collation (idempotent)."""
    if getattr(template, "_multilabel_patched", False):
        return
    template._multilabel_patched = True
    original_encode = template.encode
    original_collator = template.data_collator

    def encode_with_label_vector(inputs: Any, **kwargs: Any) -> Any:
        label_vector = inputs.get("label_vector") if isinstance(inputs, dict) else None
        encoded = original_encode(inputs, **kwargs)
        if label_vector is not None and isinstance(encoded, dict):
            encoded["label_vector"] = label_vector
        return encoded

    def collate_with_label_vector(batch: Any, **kwargs: Any) -> Any:
        result = original_collator(batch, **kwargs)
        flat_batch = sum(batch, start=[]) if batch and isinstance(batch[0], list) else batch
        label_vectors = [
            item["label_vector"]
            for item in flat_batch
            if item.get("label_vector") is not None
        ]
        if label_vectors:
            result["label_vector"] = torch.tensor(label_vectors, dtype=torch.float32)
        return result

    template.encode = encode_with_label_vector
    template.data_collator = collate_with_label_vector
