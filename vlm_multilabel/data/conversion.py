"""Shared helpers for dataset conversion to Swift JSONL records."""
from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from vlm_multilabel.constants.base import DatasetConfig


def choose_primary_label(labels: list[str]) -> str:
    """Return the first positive label, preserving the caller's canonical order."""
    if not labels:
        raise ValueError("first-positive primary selection requires at least one positive label")
    return labels[0]


def build_record(
    config: DatasetConfig,
    image_path: str,
    positive_names: Iterable[str],
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    """Build one Swift record with a first-positive primary code.

    Primary selection always follows ``config.label_codes`` order, regardless of
    annotation-file ordering. Samples without a positive label cannot provide a
    first-positive CE target and must be skipped by dataset converters.
    """
    positive_set = set(positive_names)
    ordered_positive_names = [
        name for name in config.label_names if name in positive_set
    ]
    primary_name = choose_primary_label(ordered_positive_names)
    label_dict = {name: int(name in positive_set) for name in config.label_names}
    primary_code = config.name_to_code.get(primary_name)
    if primary_code is None:
        raise ValueError(f"Positive label {primary_name!r} is not in the dataset config")
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": primary_code},
        ],
        "images": [image_path],
        "primary_code": primary_code,
        "label_vector": config.label_vector_from_labels(label_dict),
    }


def first_positive_code(
    config: DatasetConfig,
    label_vector: list[Any],
) -> str:
    """Return the canonical first-positive code for a label vector."""
    if len(label_vector) != config.num_classes:
        raise ValueError(
            f"label_vector has {len(label_vector)} entries; expected {config.num_classes}"
        )
    positive_indices = [
        index for index, value in enumerate(label_vector) if int(value) == 1
    ]
    if not positive_indices:
        raise ValueError("first-positive primary selection requires a positive label")
    return config.label_codes[positive_indices[0]]


def validate_first_positive_records(
    config: DatasetConfig,
    records: Iterable[dict[str, Any]],
    source: str = "",
    *,
    allow_no_positive: bool = False,
) -> int:
    """Validate primary codes against first-positive config order.

    Evaluation over an official multi-label test split may contain all-zero
    label vectors. These records have no first-positive CE target and must
    store ``primary_code: null``; they are accepted only when
    ``allow_no_positive`` is enabled.
    """
    prefix = f"{source}: " if source else ""
    for row, record in enumerate(records, start=1):
        label_vector = [int(value) for value in record["label_vector"]]
        if allow_no_positive and not any(label_vector):
            if record.get("primary_code") is not None:
                raise ValueError(
                    f"{prefix}record {row}: no-positive record must have "
                    "primary_code=null"
                )
            continue
        try:
            expected_code = first_positive_code(config, label_vector)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{prefix}record {row}: {exc}") from exc
        actual_code = record.get("primary_code")
        if actual_code != expected_code:
            raise ValueError(
                f"{prefix}record {row}: primary_code={actual_code!r}, "
                f"but first positive is {expected_code!r}"
            )
    return row if "row" in locals() else 0


def subsample_records(
    records: list[dict[str, Any]],
    maximum: int | None,
    seed: int,
    split_name: str,
) -> list[dict[str, Any]]:
    """Randomly downsample a split while logging the operation."""
    if maximum is None or len(records) <= maximum:
        return records
    sampled = random.Random(seed).sample(records, maximum)
    print(f"[subsample] {split_name} reduced from {len(records)} to {len(sampled)}")
    return sampled


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write records as UTF-8 JSON Lines."""
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
