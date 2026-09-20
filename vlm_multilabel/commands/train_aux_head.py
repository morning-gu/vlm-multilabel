"""Train the independent classification head (aux-head) control baseline.

Same backbone, input encoding, and readout position as the main CE+BCE
method, but supervision is a separate linear head on the last-token hidden
state trained with BCE only. No next-token CE, no MP, no BCE on the decoder
label-token logits.

Backbone adaptation options:
- LoRA (default): peft LoRA r=8/alpha=32, mirroring the main method capacity.
- --adapters <dir>: load existing LoRA adapters instead of attaching fresh
  ones (e.g. a CE-only decoder checkpoint, yielding the one-factor
  "CE(decoder) + BCE(aux-head)" cell).
- --freeze-backbone: frozen backbone (also freezes loaded adapters), trains
  the head only.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from vlm_multilabel.commands.train import configure_logging, set_seed
from vlm_multilabel.constants.base import load_config
from vlm_multilabel.data.conversion import validate_first_positive_records
from vlm_multilabel.losses.aux_head import AuxHead, aux_head_bce_loss, gather_last_hidden
from vlm_multilabel.runtime.model import (
    keep_last_logits_kwargs,
    load_jsonl,
    load_vlm,
    load_swift_template,
    patch_vision_attention,
)
from vlm_multilabel.runtime.swift_compat import (
    normalize_jsonl_media_paths,
    patch_torch_load_weights_only_fallback,
)

logger = logging.getLogger(__name__)


def forward_last_hidden(model: Any, inputs: dict[str, Any]) -> torch.Tensor:
    """Run one forward pass and return last-layer hidden states."""
    kwargs = dict(use_cache=False, output_hidden_states=True, **keep_last_logits_kwargs(model))
    try:
        outputs = model(**inputs, **kwargs)
    except TypeError:
        outputs = model(**inputs, use_cache=False, output_hidden_states=True)
    return outputs.hidden_states[-1]


def build_backbone(model: Any, freeze_backbone: bool, lora_r: int, lora_alpha: int) -> Any:
    """Attach LoRA adapters or freeze the backbone, per the selected mode."""
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        logger.info("backbone frozen; training the aux head only")
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "peft is not importable. Install it with 'pip install peft'."
        ) from exc
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def enable_gradient_checkpointing(model: Any) -> None:
    if not hasattr(model, "gradient_checkpointing_enable"):
        return
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        logger.info("gradient checkpointing enabled")
    except TypeError:
        logger.info("gradient checkpointing not supported by this model; skipped")


def encode_inputs(
    model: Any, template: Any, system_prompt: str, records: list[dict], images: list[Image.Image]
) -> dict[str, Any]:
    """Encode one batch exactly like main-method inference/evaluation."""
    from vlm_multilabel.commands.evaluate import _encode_batch

    return _encode_batch(model, template, system_prompt, records, images)


def load_adapters(model: Any, adapters: str) -> Any:
    """Wrap the base model with existing peft LoRA adapters."""
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError(
            "peft is not importable. Install it with 'pip install peft'."
        ) from exc
    model = PeftModel.from_pretrained(model, adapters)
    logger.info("loaded LoRA adapters from %s", adapters)
    return model


def run_training(args: argparse.Namespace) -> None:
    patch_torch_load_weights_only_fallback()
    configure_logging()
    set_seed(args.seed)

    config = load_config(args.dataset)
    train_data = normalize_jsonl_media_paths(args.train_data)
    records = load_jsonl(train_data)
    validate_first_positive_records(config, records, source=f"train_data={train_data}")
    logger.info("dataset=%s train_records=%d classes=%d", config.name, len(records), config.num_classes)

    model, resolved_attention = load_vlm(
        args.model, attn_implementation=args.attn_implementation
    )
    template, _ = load_swift_template(args.model, model)
    system_prompt = config.render_prompt()

    if args.adapters:
        model = load_adapters(model, args.adapters)
    else:
        model = build_backbone(model, args.freeze_backbone, args.lora_r, args.lora_alpha)
    if args.adapters and not args.freeze_backbone:
        logger.warning(
            "--adapters without --freeze-backbone continues adapter training "
            "with BCE gradients; the decoder generative space will drift from "
            "the loaded checkpoint"
        )
    if not args.freeze_backbone:
        enable_gradient_checkpointing(model)
    patch_vision_attention(model, args.attn_implementation or "sdpa")

    hidden_size = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size
    head = AuxHead(hidden_size, config.num_classes).to(model.device)
    head.float()

    lora_parameters = [p for p in model.parameters() if p.requires_grad]
    param_groups = []
    if lora_parameters:
        param_groups.append({"params": lora_parameters, "lr": args.learning_rate})
    param_groups.append({"params": head.parameters(), "lr": args.head_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)
    total_steps = max(
        1,
        int(len(records) * args.num_train_epochs)
        // (args.per_device_train_batch_size * args.gradient_accumulation_steps),
    )
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    rng = np.random.default_rng(args.seed)
    model.train()
    head.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    optimizer_step = 0
    progress = tqdm(total=total_steps, desc="aux-head train", unit="step")
    for epoch in range(int(np.ceil(args.num_train_epochs))):
        order = rng.permutation(len(records))
        for start in range(0, len(order), args.per_device_train_batch_size):
            batch_records = [records[i] for i in order[start : start + args.per_device_train_batch_size]]
            images = [Image.open(r["images"][0]).convert("RGB") for r in batch_records]
            inputs = encode_inputs(model, template, system_prompt, batch_records, images)
            _, inputs = template.pre_forward_hook(model, None, inputs)
            label_vector = torch.tensor(
                [r["label_vector"] for r in batch_records], dtype=torch.float32, device=model.device
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                hidden = forward_last_hidden(model, inputs)
            last_hidden = gather_last_hidden(hidden, inputs["attention_mask"])
            head_logits = head(last_hidden)
            loss = aux_head_bce_loss(head_logits, label_vector)
            (loss / args.gradient_accumulation_steps).backward()

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"]], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                progress.update(1)
                optimizer_step += 1
            global_step += 1
            if optimizer_step >= total_steps:
                break
        if optimizer_step >= total_steps:
            break
    progress.close()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.freeze_backbone:
        model.save_pretrained(str(output_dir))
    head.save(str(output_dir / "aux_head.pt"))
    (output_dir / "aux_head_train_args.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "dataset": args.dataset,
                "train_data": str(train_data),
                "seed": args.seed,
                "adapters": str(args.adapters or ""),
                "freeze_backbone": args.freeze_backbone,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "learning_rate": args.learning_rate,
                "head_lr": args.head_lr,
                "num_train_epochs": args.num_train_epochs,
                "hidden_size": hidden_size,
                "num_classes": config.num_classes,
                "attention": resolved_attention or "framework-default",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("aux-head training done; saved to %s", output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, choices=["voc", "chestxray14", "celeba"])
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--adapters",
        default="",
        help="existing peft LoRA adapter directory to load instead of "
        "attaching fresh LoRA (e.g. a CE-only decoder checkpoint)",
    )
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--num-train-epochs", type=float, default=2)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument(
        "--attn-implementation", default="flash_attention_2"
    )
    args = parser.parse_args(argv)
    random.seed(args.seed)
    return args


def main(argv: list[str] | None = None) -> None:
    run_training(parse_args(argv))


if __name__ == "__main__":
    main()
