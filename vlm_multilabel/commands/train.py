"""Train a VLM with the CE + MP + BCE hybrid loss for multi-label data."""
from __future__ import annotations

import argparse
import logging
import os
import random
from typing import Any

import numpy as np
import torch

from vlm_multilabel.constants.base import load_config
from vlm_multilabel.losses.trainer import make_trainer
from vlm_multilabel.losses.template_patch import patch_template_for_label_vector
from vlm_multilabel.data.conversion import validate_first_positive_records
from vlm_multilabel.runtime.model import load_jsonl, patch_vision_attention, resolve_attn_implementation
from vlm_multilabel.runtime.swift_compat import (
    customize_chat_template,
    normalize_jsonl_media_paths,
    patch_swift_no_duplicate_final_eval,
    patch_swift_symlink_none_best,
    patch_torch_load_weights_only_fallback,
)

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed set to %d", seed)


def configure_logging() -> None:
    """Reduce noisy third-party INFO logs while keeping our training logs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    for name in ("swift", "ms_swift", "datasets"):
        logging.getLogger(name).setLevel(logging.WARNING)


def build_swift_args(args: argparse.Namespace, train_data: str, val_data: str) -> list[str]:
    """Translate the public CLI into Swift's CLI representation."""
    swift_args = [
        "--model", args.model,
        "--dataset", os.path.abspath(train_data),
        "--output_dir", args.output_dir,
        "--add_version", "false",
        "--create_checkpoint_symlink", "true",
        "--seed", str(args.seed),
        "--num_train_epochs", str(args.num_train_epochs),
        "--per_device_train_batch_size", str(args.per_device_train_batch_size),
        "--per_device_eval_batch_size", str(args.per_device_eval_batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--learning_rate", str(args.learning_rate),
        "--warmup_ratio", str(args.warmup_ratio),
        "--logging_steps", str(args.logging_steps),
        "--save_strategy", "epoch",
        "--bf16", "true",
        "--report_to", "tensorboard",
        "--remove_unused_columns", "false",
        "--tuner_type", "lora",
        "--dataloader_num_workers", "4",
        "--dataset_num_proc", "2",
        "--dataloader_pin_memory", "true",
        "--group_by_length", "true",
    ]
    if args.attn_implementation:
        swift_args.extend(
            ["--model_kwargs", f'{{"attn_implementation":"{args.attn_implementation}"}}']
        )
    if val_data:
        swift_args.extend(["--val_dataset", os.path.abspath(val_data)])
    if args.adapters:
        swift_args.extend(["--adapters", os.path.abspath(args.adapters)])
    if args.resume_from_checkpoint:
        swift_args.extend(["--resume_from_checkpoint", args.resume_from_checkpoint])
    return swift_args


def run_training(args: argparse.Namespace) -> None:
    """Run a Swift SFT job with the hybrid-loss trainer."""
    try:
        from swift.pipelines.train.sft import SwiftSft
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "ms-swift is not importable. Install it with 'pip install \"ms-swift[llm]\"'."
        ) from exc

    patch_torch_load_weights_only_fallback()
    patch_swift_no_duplicate_final_eval()
    patch_swift_symlink_none_best()
    configure_logging()
    set_seed(args.seed)

    config = load_config(args.dataset)
    requested_attention = args.attn_implementation
    args.attn_implementation = resolve_attn_implementation(
        args.model, requested_attention
    ) if requested_attention else ""
    logger.info(
        "dataset=%s classes=%d attention=%s",
        config.name,
        config.num_classes,
        args.attn_implementation or "framework-default",
    )

    train_data = normalize_jsonl_media_paths(args.train_data)
    val_data = normalize_jsonl_media_paths(args.val_data) if args.val_data else ""
    validate_first_positive_records(
        config, load_jsonl(train_data), source=f"train_data={train_data}"
    )
    if val_data:
        validate_first_positive_records(
            config, load_jsonl(val_data), source=f"val_data={val_data}"
        )
    logger.info("validated first-positive primary labels")
    swift_args = build_swift_args(args, train_data, val_data)

    class MultiLabelSft(SwiftSft):
        """Swift SFT adapter that installs multi-label behavior at run time."""

        def run(self) -> Any:
            configure_logging()
            patch_template_for_label_vector(self.template)
            customize_chat_template(self.template.tokenizer)
            label_token_ids = config.get_label_token_ids(self.template.tokenizer)
            logger.info(
                "single-token label IDs: %d/%d",
                len(label_token_ids),
                config.num_classes,
            )
            missing = [
                code for code in config.label_codes if code not in label_token_ids
            ]
            if missing:
                logger.warning("BCE will skip non-single-token label codes: %s", missing)

            swift_config = self.args
            train_dataset, eval_dataset = self._prepare_dataset()
            if swift_config.task_type == "seq_cls":
                swift_config.problem_type = swift_config.problem_type or getattr(
                    self.model.config, "problem_type", None
                )
            swift_config.save_args()
            self.model = self.prepare_model(
                swift_config,
                self.model,
                template=self.template,
                train_dataset=train_dataset,
            )
            patch_vision_attention(self.model, args.attn_implementation or "sdpa")
            trainer_class = make_trainer(
                label_token_ids,
                bce_weight=args.bce_weight,
                mp_weight=args.mp_weight,
            )
            trainer = trainer_class(
                model=self.model,
                args=swift_config.training_args,
                template=self.template,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )
            return self.train(trainer)

    MultiLabelSft(swift_args).main()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="base model id/path")
    parser.add_argument(
        "--dataset", required=True, choices=["voc", "chestxray14", "celeba"]
    )
    parser.add_argument("--train-data", default="data/{dataset}/train.jsonl")
    parser.add_argument("--val-data", default="data/{dataset}/val.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/lora")
    parser.add_argument("--adapters", default="")
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--mp-weight", type=float, default=1.0)
    parser.add_argument("--num-train-epochs", type=float, default=2)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--attn-implementation", default="flash_attention_2"
    )
    args = parser.parse_args(argv)
    args.train_data = args.train_data.replace("{dataset}", args.dataset)
    args.val_data = args.val_data.replace("{dataset}", args.dataset)
    return args


def main(argv: list[str] | None = None) -> None:
    run_training(parse_args(argv))


if __name__ == "__main__":
    main()
