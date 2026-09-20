"""Unified command-line entry point for vlm-multilabel."""
from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from importlib import import_module
from typing import Any

_COMMAND_MODULES = {
    "train": "vlm_multilabel.commands.train",
    "evaluate": "vlm_multilabel.commands.evaluate",
    "train-aux-head": "vlm_multilabel.commands.train_aux_head",
    "evaluate-aux-head": "vlm_multilabel.commands.evaluate_aux_head",
    "posthoc-calibration": "vlm_multilabel.commands.posthoc_calibration",
    "infer": "vlm_multilabel.commands.infer",
    "convert": None,
    "resize-images": "vlm_multilabel.commands.resize_images",
    "verify-theory": "vlm_multilabel.commands.verify_theory",
}

_DATASETS = {
    "voc": "vlm_multilabel.commands.convert_voc",
    "chestxray14": "vlm_multilabel.commands.convert_chestxray14",
    "celeba": "vlm_multilabel.commands.convert_celeba",
}


def _command_main(module_name: str) -> Callable[..., Any]:
    module = import_module(module_name)
    return module.main


def _dispatch_convert(command_args: list[str]) -> None:
    """Route ``convert`` arguments to the selected dataset converter."""
    dataset: str | None = None
    if command_args and command_args[0] in _DATASETS:
        dataset = command_args.pop(0)
    else:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--dataset", required=True, choices=sorted(_DATASETS))
        args, command_args = parser.parse_known_args(command_args)
        dataset = args.dataset

    _command_main(_DATASETS[dataset])(command_args)


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch to a dataset converter or one of the top-level commands."""
    parser = argparse.ArgumentParser(
        prog="vlm-multilabel",
        description="Train, evaluate, convert, and inspect multi-label VLM datasets.",
    )
    parser.add_argument(
        "command",
        choices=sorted(_COMMAND_MODULES),
        help="command to run; append --help after the command for details",
    )
    parser.add_argument("command_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if args.command == "convert":
        _dispatch_convert(args.command_args)
        return

    _command_main(_COMMAND_MODULES[args.command])(args.command_args)


if __name__ == "__main__":
    main()
