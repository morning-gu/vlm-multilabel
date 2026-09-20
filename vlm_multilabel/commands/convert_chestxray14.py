"""Convert ChestX-ray14 (NIH) annotations to ms-swift jsonl format.

Reads Data_Entry_2017.csv and produces train/val/test jsonl with
label_vector (14-dim 0/1). The 14 classes are the standard thoracic
disease classes per Wang et al. (CVPR 2017), Hernia included. ``No
Finding`` is represented by an all-zero 14-dim label vector.

Training/validation records follow the first-positive primary protocol and
skip no-positive records, which cannot provide a CE target. The official
test split is preserved in full (including no-positive records) for
multi-label sigmoid metrics.

Data format: "Finding Labels" column is pipe-separated, e.g.
"Atelectasis|Effusion" or "Hernia".

Official split files (train_val_list.txt, test_list.txt) determine
the train+val vs test boundary. train_val is further split 90/10
into train/val with a fixed seed.

Usage:
  python -m scripts.convert_chestxray14_to_swift \\
      --csv /data/ChestX-ray14/Data_Entry_2017.csv \\
      --image-root /data/ChestX-ray14/images \\
      --split-dir /data/ChestX-ray14 \\
      --out data/chestxray14 \\
      --max-train-samples 30000
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from vlm_multilabel.constants.chestxray14 import ChestXray14Config

from vlm_multilabel.data.images import copy_images_to_outdir, parse_image_size
from vlm_multilabel.data.conversion import build_record, subsample_records, write_jsonl

# ChestX-ray14 finding labels that map to our config classes.
# All 14 standard pathology classes from Wang et al. (CVPR 2017).
FINDING_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural Thickening", "Hernia",
]

NO_FINDING_LABEL = "no finding"

def parse_finding_labels(labels_str: str) -> list[str]:
    """Parse pipe-separated finding labels into a list of label names."""
    return [s.strip() for s in labels_str.split("|") if s.strip()]


def normalize_finding_label(label: str) -> str:
    """Normalize NIH spelling variants such as ``Pleural_Thickening``."""
    return " ".join(label.strip().replace("_", " ").casefold().split())


def to_record(
    config: ChestXray14Config,
    image_path: str,
    finding_labels_str: str,
    sys_prompt: str,
    *,
    allow_no_positive: bool = False,
) -> dict | None:
    """Build a ChestX-ray14 record.

    By default, return ``None`` for no-positive records so they cannot enter
    first-positive CE training. With ``allow_no_positive=True`` (used for the
    official test split), return an all-zero multi-label vector and a null
    primary code.
    """
    normalized_to_canonical = {
        normalize_finding_label(name): name for name in config.label_names
    }
    positives: list[str] = []
    unknown_labels: list[str] = []
    for raw_label in parse_finding_labels(finding_labels_str):
        normalized = normalize_finding_label(raw_label)
        if normalized == NO_FINDING_LABEL:
            continue
        canonical = normalized_to_canonical.get(normalized)
        if canonical is None:
            unknown_labels.append(raw_label)
        elif canonical not in positives:
            positives.append(canonical)
    if unknown_labels:
        raise ValueError(
            f"Unknown ChestX-ray14 finding labels: {unknown_labels}; "
            f"image={image_path!r}"
        )
    if not positives:
        if not allow_no_positive:
            return None
        return {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": config.user_prompt},
            ],
            "images": [image_path],
            "primary_code": None,
            "label_vector": config.label_vector_from_labels({}),
        }

    return build_record(
        config,
        image_path,
        positives,
        sys_prompt,
        config.user_prompt,
    )

def read_split_file(path: str) -> set:
    """Read a split file (one image filename per line) into a set."""
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True,
                    help="Data_Entry_2017.csv path")
    ap.add_argument("--image-root", required=True,
                    help="ChestX-ray14 image directory (contains .png files)")
    ap.add_argument("--split-dir", default=None,
                    help="Directory with train_val_list.txt and test_list.txt "
                         "(default: same as --csv's parent)")
    ap.add_argument("--out", default="data/chestxray14",
                    help="output directory")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="fraction of train_val for validation (when split files exist)")
    ap.add_argument("--max-train-samples", type=int, default=None,
                    help="if set, randomly subsample train split to at most N records")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--primary-strategy", default="first", choices=["first"],
                    help="fixed first-positive primary selection in config order")
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

    config = ChestXray14Config()
    sys_prompt = config.render_prompt()

    image_root = Path(a.image_root)
    split_dir = Path(a.split_dir) if a.split_dir else Path(a.csv).parent
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # Read official split files if they exist
    train_val_file = split_dir / "train_val_list.txt"
    test_file = split_dir / "test_list.txt"
    has_official_splits = train_val_file.exists() and test_file.exists()

    if has_official_splits:
        train_val_set = read_split_file(str(train_val_file))
        test_set = read_split_file(str(test_file))
        print(f"[split] official splits: train_val={len(train_val_set)}, test={len(test_set)}")
    else:
        train_val_set = None
        test_set = None
        print(f"[split] no official split files found in {split_dir}; "
              f"using --val-frac={a.val_frac} for train/val, no test split")

    # Build records, partitioned by split
    train_val_records = []
    test_records = []
    unsplit_records = []
    assigned_train_val: set[str] = set()
    assigned_test: set[str] = set()

    skipped_no_positive = 0
    test_no_positive = 0
    seen_images: set[str] = set()
    with open(a.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row.get("Image Index", "").strip()
            if not img_name:
                continue
            if img_name in seen_images:
                raise ValueError(f"Duplicate ChestX-ray14 CSV row for image: {img_name}")
            seen_images.add(img_name)
            img_path = (image_root / img_name).as_posix()
            if not Path(img_path).exists():
                raise FileNotFoundError(f"Missing ChestX-ray14 image: {img_path}")

            finding_labels_str = row.get("Finding Labels", "")
            if has_official_splits:
                if img_name in train_val_set:
                    assigned_train_val.add(img_name)
                    rec = to_record(config, img_path, finding_labels_str, sys_prompt)
                    if rec is None:
                        skipped_no_positive += 1
                        continue
                    train_val_records.append(rec)
                elif img_name in test_set:
                    assigned_test.add(img_name)
                    rec = to_record(
                        config, img_path, finding_labels_str, sys_prompt,
                        allow_no_positive=True,
                    )
                    test_records.append(rec)
                else:
                    raise ValueError(
                        f"ChestX-ray14 image is not in an official split file: {img_name}"
                    )
            else:
                rec = to_record(config, img_path, finding_labels_str, sys_prompt)
                if rec is None:
                    skipped_no_positive += 1
                    continue
                unsplit_records.append(rec)

    if skipped_no_positive:
        print(
            "[chestxray14] skipped "
            f"{skipped_no_positive} train/val records with no positive finding"
        )
    if has_official_splits:
        test_no_positive = sum(
            sum(record["label_vector"]) == 0 for record in test_records
        )
        print(
            "[chestxray14] retained "
            f"{test_no_positive} no-positive official-test records"
        )

    if has_official_splits:
        missing_from_csv = test_set - assigned_test
        if missing_from_csv:
            preview = ", ".join(sorted(missing_from_csv)[:10])
            raise ValueError(
                f"{len(missing_from_csv)} official-test images are absent from CSV: "
                f"{preview}"
            )
        missing_from_csv = train_val_set - assigned_train_val
        if missing_from_csv:
            preview = ", ".join(sorted(missing_from_csv)[:10])
            raise ValueError(
                f"{len(missing_from_csv)} official train_val images are absent from CSV: "
                f"{preview}"
            )

    # Split into train/val
    random.seed(a.seed)
    if has_official_splits:
        random.shuffle(train_val_records)
        n_val = int(len(train_val_records) * a.val_frac)
        val_records = train_val_records[:n_val]
        train_records = train_val_records[n_val:]
    else:
        random.shuffle(unsplit_records)
        n_val = int(len(unsplit_records) * a.val_frac)
        val_records = unsplit_records[:n_val]
        train_records = unsplit_records[n_val:]
        test_records = []  # no test split without official files

    train_records = subsample_records(
        train_records, a.max_train_samples, a.seed, "train"
    )
    val_records = subsample_records(val_records, a.max_val_samples, a.seed + 1, "val")

    # Copy or resize images to <out>/images/ if requested
    image_size = parse_image_size(a.image_size) if a.image_size else None
    if a.copy_images or image_size is not None:
        all_recs = train_records + val_records + test_records
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

    for split_name, split_records in [("train", train_records),
                                       ("val", val_records),
                                       ("test", test_records)]:
        if not split_records:
            continue
        out_file = out / f"{split_name}.jsonl"
        write_jsonl(out_file, split_records)
        print(f"[{split_name}] {len(split_records):>6d} -> {out_file}")

    total = len(train_records) + len(val_records) + len(test_records)
    print(f"\ntotal={total}  train={len(train_records)}  "
          f"val={len(val_records)}  test={len(test_records)}")


if __name__ == "__main__":
    main()
