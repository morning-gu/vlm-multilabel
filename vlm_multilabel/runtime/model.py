"""Model-loading and runtime helpers shared by VLM scripts."""
from __future__ import annotations

import inspect
import json
import logging

import numpy as np
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_FLASH_ATTENTION_HEAD_DIM = 256


def keep_last_logits_kwargs(model: Any) -> dict[str, int]:
    """Kwargs that restrict the model's internal lm_head call to position -1.

    transformers renamed ``num_logits_to_keep`` to ``logits_to_keep``; recent
    versions silently swallow the old name through ``**kwargs`` and then
    compute full-sequence logits, which on large-vocabulary models is a large
    transient allocation (an OOM hazard at inference batch sizes). Probing the
    forward signature picks whichever name the installed version honors; when
    neither is declared (very old versions) the empty dict preserves the
    historical full-logits behavior.
    """
    candidates = ("logits_to_keep", "num_logits_to_keep")
    for target in (model, getattr(model, "get_base_model", lambda: None)()):
        forward = getattr(target, "forward", None)
        if forward is None:
            continue
        try:
            parameters = inspect.signature(forward).parameters
        except (TypeError, ValueError):
            continue
        for name in candidates:
            if name in parameters:
                return {name: 1}
    return {}


def resolve_attn_implementation(model_path: str, requested: str) -> str:
    """Downgrade FlashAttention-2 to SDPA when the model head_dim is too large."""
    if requested != "flash_attention_2":
        return requested
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return requested
    try:
        config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read model config at %s; using requested attention", config_path)
        return requested

    text_config = config.get("text_config", config)
    head_dim = text_config.get("head_dim")
    if head_dim is None:
        hidden_size = text_config.get("hidden_size", 0)
        num_heads = text_config.get("num_attention_heads", 0)
        if hidden_size and num_heads:
            head_dim = hidden_size // num_heads
    if head_dim and head_dim > MAX_FLASH_ATTENTION_HEAD_DIM:
        logger.info(
            "head_dim=%s exceeds FlashAttention-2 limit; falling back to sdpa",
            head_dim,
        )
        return "sdpa"
    return requested


def patch_vision_attention(model: Any, attn_implementation: str = "sdpa") -> int:
    """Patch VLM submodules that incorrectly retained eager attention."""
    base_model = model
    for attribute in ("base_model", "model"):
        candidate = getattr(base_model, attribute, None)
        if candidate is not None and hasattr(candidate, "model"):
            base_model = candidate
            break

    patched = 0
    for _, module in base_model.named_modules():
        module_attention = getattr(module, "_attn_implementation", None)
        if module_attention == "eager":
            module._attn_implementation = attn_implementation
            patched += 1
        config = getattr(module, "config", None)
        if config is not None and getattr(config, "_attn_implementation", None) == "eager":
            config._attn_implementation = attn_implementation
    if patched:
        logger.info("Patched %d eager attention modules to %s", patched, attn_implementation)
    return patched


def load_vlm(
    model_id: str,
    adapters: str = "",
    attn_implementation: str = "",
):
    """Load an image-to-text VLM and attach optional Swift LoRA adapters."""
    from transformers import AutoModelForImageTextToText

    resolved = resolve_attn_implementation(model_id, attn_implementation) if attn_implementation else ""
    model_kwargs = dict(torch_dtype="auto", device_map="auto", trust_remote_code=True)
    if resolved:
        model_kwargs["attn_implementation"] = resolved
    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    if adapters:
        from swift.tuners import Swift
        model = Swift.from_pretrained(model, adapters)
    model.eval()
    patch_vision_attention(model, resolved or "sdpa")
    return model, resolved


def load_swift_template(model_id: str, model: Any):
    """Load a left-padded Swift inference template compatible with the model."""
    from swift import get_processor, get_template

    processor = get_processor(model_id)
    template = get_template(processor, enable_thinking=False, padding_side="left")
    if getattr(template, "use_model", False):
        template.model = model
    return template, processor.tokenizer


def load_jsonl(path: str) -> list[dict]:
    """Load a JSON-lines file into a list of dictionaries."""
    records: list[dict] = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object on {path}:{line_number}")
            records.append(value)
    return records


def build_label_matrix(records: list[dict], num_classes: int) -> "np.ndarray":
    """Build a [num_records, num_classes] 0/1 matrix from `label_vector`."""
    import numpy as np

    labels = np.zeros((len(records), num_classes), dtype=np.float32)
    for row, record in enumerate(records):
        vector = record.get("label_vector", [])
        for column, value in enumerate(vector[:num_classes]):
            if value:
                labels[row, column] = 1.0
    return labels
