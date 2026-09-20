"""Convert CelebA annotations (CSV) to ms-swift jsonl format.

Reads list_attr_celeba.csv and list_eval_partition.csv to produce
train/val/test jsonl with a 40-dim label_vector (0/1) aligned to the
config's label_codes order, plus a single primary_code for the CE loss.

Usage:
  python -m scripts.convert_celeba_to_swift --root /data/CelebA --out data/celeba

--root points at the dataset dir holding img_align_celeba/ and the list_*.csv
files; image paths in the output jsonl point there. --attr/--partition/
--image-root override the derived paths when needed.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from vlm_multilabel.constants.celeba import CelebAConfig

from vlm_multilabel.data.images import copy_images_to_outdir, parse_image_size
from vlm_multilabel.data.conversion import build_record, subsample_records, write_jsonl

# Official CelebA partition ids: 0=train, 1=val, 2=test.
SPLIT_BY_ID = {0: "train", 1: "val", 2: "test"}


def to_record(
    config: CelebAConfig,
    image_path: str,
    attr_row: dict[str, str],
    sys_prompt: str,
) -> dict | None:
    """Build a CelebA record, or return ``None`` when no positive label exists."""
    positives = [
        name for name in config.label_names
        if int(attr_row.get(name, "-1")) > 0
    ]
    if not positives:
        return None
    return build_record(
        config,
        image_path,
        positives,
        sys_prompt,
        config.user_prompt,
    )

def read_partitions(path: str) -> dict:
    """Read list_eval_partition.csv -> {image_id: split_id}. CSV has a header
    row (image_id,partition), so DictReader handles it; no line to skip."""
    partitions = {}
    with open(path, encoding="utf-8-sig", newline="") as pf:
        for row in csv.DictReader(pf):
            img = row["image_id"].strip()
            partitions[img] = int(row["partition"].strip())
    return partitions


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None,
                    help="CelebA dataset root dir (holds img_align_celeba/ and "
                         "list_attr_celeba.csv / list_eval_partition.csv). When set, "
                         "--attr/--partition/--image-root default to paths under it; "
                         "image paths in the output jsonl point here.")
    ap.add_argument("--attr", default=None,
                    help="list_attr_celeba.csv path (default <root>/list_attr_celeba.csv)")
    ap.add_argument("--partition", default=None,
                    help="list_eval_partition.csv path "
                         "(default <root>/list_eval_partition.csv)")
    ap.add_argument("--image-root", default=None,
                    help="img_align_celeba dir path (default <root>/img_align_celeba)")
    ap.add_argument("--out", default="data/celeba", help="output directory")
    ap.add_argument("--primary-strategy", default="first", choices=["first"],
                    help="fixed first-positive primary selection in config order")
    ap.add_argument("--max-train-samples", type=int, default=None,
                    help="if set, randomly subsample train split to at most N records")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for train/val subsampling")
    ap.add_argument("--max-val-samples", type=int, default=None,
                    help="if set, randomly subsample val split to at most N records")
    ap.add_argument("--copy-images", action="store_true",
                    help="copy image files to <out>/images/ and rewrite paths")
    ap.add_argument("--image-size", default=None,
                    help="resize images to this size during copy, e.g. '448' "
                         "(square) or '448x448'; implies --copy-images")
    ap.add_argument("--no-pad", action="store_true",
                    help="force-resize without maintaining aspect ratio "
                         "(default: pad to square with black borders)")
    a = ap.parse_args(argv)

    root = Path(a.root) if a.root else None
    attr_path = a.attr or (str(root / "list_attr_celeba.csv") if root else None)
    partition_path = (a.partition
                      or (str(root / "list_eval_partition.csv") if root else ""))
    image_root_str = (a.image_root
                      or (str(root / "img_align_celeba") if root else ""))
    if not attr_path or not Path(attr_path).is_file():
        ap.error("--attr not found; pass --root or --attr explicitly")
    if not image_root_str:
        ap.error("--image-root not set; pass --root or --image-root explicitly")
    image_root = Path(image_root_str)
    if not image_root.exists():
        print(f"[celeba] WARNING: image-root {image_root} does not exist; "
              f"jsonl paths recorded but images not readable yet")

    config = CelebAConfig()
    sys_prompt = config.render_prompt()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # Read partition file (CSV: image_id,partition; 0=train/1=val/2=test).
    partitions = {}
    if partition_path and Path(partition_path).exists():
        partitions = read_partitions(partition_path)
        print(f"[celeba] {len(partitions)} partition entries")

    split_records = {"train": [], "val": [], "test": []}

    # list_attr_celeba.csv: header = image_id,<40 attribute names>; rows =
    # image_id,attr1,...,attr40 with -1/1 values. utf-8-sig tolerates a BOM.
    with open(attr_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        attr_cols = reader.fieldnames[1:] if reader.fieldnames else []
        print(f"[celeba] {len(attr_cols)} attribute columns")

        # Warn if the CSV attribute set drifted from the config (order aside).
        file_names = set(attr_cols)
        config_names = set(config.label_names)
        if file_names != config_names:
            print(f"[celeba] WARNING attribute mismatch: "
                  f"missing={config_names - file_names} "
                  f"extra={file_names - config_names}")

        skipped_no_positive = 0
        for row in reader:
            img_name = row["image_id"].strip()
            img_path = (image_root / img_name).as_posix()
            rec = to_record(config, img_path, row, sys_prompt)
            if rec is None:
                skipped_no_positive += 1
                continue

            split_id = partitions.get(img_name, 0) if partitions else 0
            split_name = SPLIT_BY_ID.get(split_id, "train")
            split_records[split_name].append(rec)

        if skipped_no_positive:
            print(f"[celeba] skipped {skipped_no_positive} records with no positive label")

    split_records["train"] = subsample_records(
        split_records["train"], a.max_train_samples, a.seed, "train"
    )
    split_records["val"] = subsample_records(
        split_records.get("val", []), a.max_val_samples, a.seed + 1, "val"
    )

    # Copy or resize images to <out>/images/ if requested
    image_size = parse_image_size(a.image_size) if a.image_size else None
    if a.copy_images or image_size is not None:
        all_recs = []
        for recs in split_records.values():
            all_recs.extend(recs)
        images_subdir = "images"
        if image_size is not None:
            if isinstance(image_size, int):
                images_subdir = f"images_{image_size}"
            else:
                images_subdir = f"images_{image_size[0]}x{image_size[1]}"
        copy_images_to_outdir(
            all_recs, out,
            images_subdir=images_subdir,
            image_size=image_size,
            pad_to_square=not a.no_pad,
        )

    for split_name, records in split_records.items():
        if not records:
            continue
        out_file = out / f"{split_name}.jsonl"
        write_jsonl(out_file, records)
        print(f"[{split_name}] {len(records):>6d} -> {out_file}")

    total = sum(len(r) for r in split_records.values())
    print(f"\ntotal={total}  "
          f"train={len(split_records['train'])}  "
          f"val={len(split_records['val'])}  "
          f"test={len(split_records['test'])}")


if __name__ == "__main__":
    main()
