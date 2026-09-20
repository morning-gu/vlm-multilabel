"""Infer multi-label predictions directly from images."""
from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from vlm_multilabel.constants.base import DatasetConfig, load_config
from vlm_multilabel.runtime.model import load_swift_template, load_vlm

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".gif"
}


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    template: Any
    tokenizer: Any
    config: DatasetConfig
    label_token_ids: dict[str, int]

    @property
    def token_ids(self) -> list[int]:
        return list(self.label_token_ids.values())

    @property
    def token_to_code(self) -> dict[int, str]:
        return {token_id: code for code, token_id in self.label_token_ids.items()}


def collect_images(args: argparse.Namespace) -> list[Path]:
    """Collect unique supported image paths from CLI sources."""
    paths: list[Path] = []
    seen: set[str] = set()

    def add(candidate: str | Path) -> None:
        path = Path(candidate).resolve()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            key = str(path)
            if key not in seen:
                seen.add(key)
                paths.append(path)

    for image_path in args.images or []:
        add(image_path)
    if args.image_dir:
        directory = Path(args.image_dir)
        if not directory.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {directory}")
        for path in sorted(directory.rglob("*")):
            add(path)
    if args.glob:
        import glob
        for path in glob.glob(args.glob, recursive=True):
            add(path)
    return paths


def load_model_bundle(args: argparse.Namespace) -> ModelBundle:
    config = load_config(args.dataset)
    model, _ = load_vlm(
        args.model,
        adapters=args.adapters,
        attn_implementation=args.attn_implementation,
    )
    template, tokenizer = load_swift_template(args.model, model)
    torch.set_float32_matmul_precision("high")
    label_token_ids = config.get_label_token_ids(tokenizer)
    return ModelBundle(model, template, tokenizer, config, label_token_ids)


def make_prediction_record(
    config: DatasetConfig,
    token_to_code: dict[int, str],
    image_path: Path,
    probabilities: np.ndarray,
    primary_token_id: int,
    threshold: float,
    top_k: int,
) -> dict[str, Any]:
    code_probabilities = {
        token_to_code[token_id]: float(probability)
        for token_id, probability in zip(token_to_code.keys(), probabilities, strict=True)
    }
    positives = sorted(
        (
            (code, config.code_to_name.get(code, code), probability)
            for code, probability in code_probabilities.items()
            if probability >= threshold
        ),
        key=lambda item: item[2],
        reverse=True,
    )
    top_labels = sorted(
        code_probabilities.items(), key=lambda item: item[1], reverse=True
    )[:top_k]
    primary_code = token_to_code[primary_token_id]
    return {
        "image": str(image_path),
        "primary_code": primary_code,
        "primary_name": config.code_to_name.get(primary_code, primary_code),
        "probabilities": code_probabilities,
        "positive_labels": [
            {"code": code, "name": name, "probability": probability}
            for code, name, probability in positives
        ],
        "top_k": [
            {"code": code, "name": config.code_to_name.get(code, code), "probability": probability}
            for code, probability in top_labels
        ],
    }


def run_inference(args: argparse.Namespace, bundle: ModelBundle, image_paths: list[Path]) -> list[dict[str, Any]]:
    """Run batched single-forward-pass inference over collected images."""
    from swift import InferRequest

    token_ids = bundle.token_ids
    token_id_set = set(token_ids)
    token_id_tensor = torch.tensor(token_ids, device=bundle.model.device, dtype=torch.long)
    results: list[dict[str, Any]] = []

    def load_images(paths: list[Path]) -> list[Image.Image]:
        return [Image.open(path).convert("RGB") for path in paths]

    with ThreadPoolExecutor(max_workers=2) as image_pool:
        batch_starts = list(range(0, len(image_paths), args.batch_size))
        future = image_pool.submit(load_images, image_paths[: args.batch_size])
        for batch_index, start in enumerate(batch_starts):
            batch_paths = image_paths[start : start + args.batch_size]
            images = future.result()
            if batch_index + 1 < len(batch_starts):
                next_start = batch_starts[batch_index + 1]
                future = image_pool.submit(
                    load_images, image_paths[next_start : next_start + args.batch_size]
                )
            requests = [
                InferRequest(
                    messages=[
                        {"role": "system", "content": bundle.config.render_prompt()},
                        {"role": "user", "content": "<image>"},
                    ],
                    images=[image],
                )
                for image in images
            ]
            encoded = [
                bundle.template.encode(request, return_template_inputs=True)
                for request in requests
            ]
            for item in encoded:
                item.pop("template_inputs", None)
            inputs = bundle.template.data_collator(encoded)
            inputs = {
                key: value.to(bundle.model.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            _, inputs = bundle.template.pre_forward_hook(bundle.model, None, inputs)

            with torch.inference_mode():
                outputs = bundle.model(**inputs, use_cache=False, num_logits_to_keep=1)
                last_logits = outputs.logits[:, -1, :]
                predicted_tokens = last_logits.argmax(dim=-1).tolist()
                label_logits = last_logits[:, token_id_tensor]
                batch_probabilities = torch.sigmoid(label_logits.float()).cpu().numpy()

            for row, (image_path, predicted_token, probabilities) in enumerate(
                zip(batch_paths, predicted_tokens, batch_probabilities, strict=True)
            ):
                if predicted_token not in token_id_set:
                    predicted_token = token_ids[int(np.argmax(probabilities))]
                results.append(
                    make_prediction_record(
                        bundle.config,
                        bundle.token_to_code,
                        image_path,
                        probabilities,
                        predicted_token,
                        args.threshold,
                        args.top_k,
                    )
                )
            processed = min(start + args.batch_size, len(image_paths))
            if batch_index % 10 == 0 or processed == len(image_paths):
                logger.info("inferred %d/%d images", processed, len(image_paths))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return results


def print_results(results: list[dict[str, Any]], threshold: float, top_k: int) -> None:
    for result in results:
        logger.info("image=%s primary=%s (%s)", result["image"], result["primary_code"], result["primary_name"])
        if result["positive_labels"]:
            for item in result["positive_labels"]:
                logger.info("positive %s (%s): %.4f", item["code"], item["name"], item["probability"])
        else:
            logger.info("no labels above threshold %.3f", threshold)
        for item in result["top_k"]:
            logger.info("top-%d %s (%s): %.4f", top_k, item["code"], item["name"], item["probability"])


def save_results(results: list[dict[str, Any]], output: str) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("results saved to %s", output_path)


def infer(args: argparse.Namespace) -> list[dict[str, Any]]:
    image_paths = collect_images(args)
    if not image_paths:
        logger.warning("No supported images found")
        return []
    logger.info("Found %d images", len(image_paths))
    bundle = load_model_bundle(args)
    results = run_inference(args, bundle, image_paths)
    print_results(results, args.threshold, args.top_k)
    if args.output:
        save_results(results, args.output)
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapters", default="")
    parser.add_argument("--dataset", required=True, choices=["voc", "chestxray14", "celeba"])
    parser.add_argument("--images", nargs="+", default=[])
    parser.add_argument("--image-dir", default="")
    parser.add_argument("--glob", default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    args = parser.parse_args(argv)
    if not args.images and not args.image_dir and not args.glob:
        parser.error("provide at least one of --images, --image-dir, --glob")
    return args


def main(argv: list[str] | None = None) -> None:
    infer(parse_args(argv))


if __name__ == "__main__":
    main()
