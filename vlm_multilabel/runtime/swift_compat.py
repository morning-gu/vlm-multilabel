"""Compatibility patches and JSONL normalization for ms-swift training."""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def patch_torch_load_weights_only_fallback() -> None:
    """Retry trusted checkpoint loading once when PyTorch rejects NumPy globals."""
    import torch

    original_load = torch.load

    def compatible_load(*args: Any, **kwargs: Any) -> Any:
        try:
            return original_load(*args, **kwargs)
        except pickle.UnpicklingError:
            if kwargs.get("weights_only") is not False:
                logger.warning("Retrying torch.load with weights_only=False")
                kwargs["weights_only"] = False
                return original_load(*args, **kwargs)
            raise

    torch.load = compatible_load


def patch_swift_no_duplicate_final_eval() -> None:
    """Suppress Swift's redundant forced evaluation after the final epoch."""
    try:
        import swift.trainers.patcher as swift_patcher
        from transformers.trainer_utils import IntervalStrategy
    except (ImportError, AttributeError) as exc:
        logger.warning("Swift duplicate-evaluation patch skipped: %s", exc)
        return

    flow_callback = getattr(swift_patcher, "DefaultFlowCallbackNew", None)
    if flow_callback is None or not hasattr(flow_callback, "on_step_end"):
        logger.warning("Swift duplicate-evaluation patch skipped: callback API changed")
        return
    original_on_step_end = flow_callback.on_step_end

    def on_step_end_without_final_eval(self: Any, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        control = original_on_step_end(self, args, state, control, **kwargs)
        strategy = getattr(args, "eval_strategy", None) or getattr(
            args, "evaluation_strategy", None
        )
        if state.global_step == state.max_steps and strategy == IntervalStrategy.EPOCH:
            control.should_evaluate = False
        return control

    flow_callback.on_step_end = on_step_end_without_final_eval
    logger.info("Patched Swift to skip redundant final-step evaluation")


def patch_swift_symlink_none_best() -> None:
    """Fall back to last checkpoint when Swift has no best_model_checkpoint.

    Swift's ``_handle_trainer_state`` calls ``os.symlink(state.best_model_checkpoint, ...)``
    unconditionally.  Without a validation metric (``load_best_model_at_end`` /
    ``metric_for_best_model``) the field stays ``None`` and the symlink call
    raises ``TypeError: symlink: src should be string, bytes or os.PathLike, not NoneType``.
    Point the ``best`` link at the last checkpoint instead so training can finish.
    """
    try:
        from swift.pipelines.train.sft import SwiftSft
    except (ImportError, AttributeError) as exc:
        logger.warning("Swift best-symlink patch skipped: %s", exc)
        return

    original_handle = SwiftSft._handle_trainer_state

    def _handle_trainer_state(self: Any, trainer: Any, is_write_rank: bool) -> Any:
        state = trainer.state
        if getattr(state, "best_model_checkpoint", None) is None:
            last = getattr(state, "last_model_checkpoint", None)
            if last:
                state.best_model_checkpoint = last
        return original_handle(self, trainer, is_write_rank)

    SwiftSft._handle_trainer_state = _handle_trainer_state
    logger.info("Patched Swift best-checkpoint symlink to fall back to last checkpoint")


def customize_chat_template(tokenizer: Any) -> None:
    """Default `enable_thinking` to False for Qwen-style chat templates."""
    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        return
    line = "{% set enable_thinking = enable_thinking | default(false) %}"
    if line in chat_template.splitlines()[0]:
        return
    tokenizer.chat_template = f"{line}\n{chat_template}"
    logger.info("Set chat-template enable_thinking default to False")


def normalize_jsonl_media_paths(path: str) -> str:
    """Normalize Windows separators in local media paths by writing a fixed copy."""
    source = Path(path)
    if not source.is_file():
        return path

    with source.open(encoding="utf-8") as stream:
        needs_fix = any(
            isinstance(media_path, str) and "\\" in media_path
            for line in stream
            if line.strip()
            for media_path in json.loads(line).get("images", [])
        )
    if not needs_fix:
        return path

    fixed = source.with_name(f"{source.stem}_fixed{source.suffix}")
    logger.info("Normalizing media paths: %s -> %s", source, fixed)
    with source.open(encoding="utf-8") as input_stream, fixed.open(
        "w", encoding="utf-8"
    ) as output_stream:
        for line in input_stream:
            if not line.strip():
                continue
            record = json.loads(line)
            for media_field in ("images", "videos", "audios"):
                media_paths = record.get(media_field)
                if media_paths:
                    record[media_field] = [
                        item.replace("\\", "/") if isinstance(item, str) else item
                        for item in media_paths
                    ]
            output_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(fixed)
