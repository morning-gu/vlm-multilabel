"""Convert PASCAL VOC annotations to ms-swift jsonl format.

Reads VOC XML annotation files and produces jsonl with label_vector
(20-dim 0/1), primary_code, and messages fields.

Usage:
  # VOC 2012 (train/val)
  python -m scripts.convert_voc_to_swift \\
      --voc-root /data/VOCdevkit/VOC2012 --out data/voc

  # VOC 2007 standard multi-label protocol (trainval -> train, test -> test)
  python -m scripts.convert_voc_to_swift \\
      --voc-root /data/VOCdevkit/VOC2007 --out data/voc2007 \\
      --splits train:trainval,test:test
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from vlm_multilabel.constants.voc import VOCConfig

from vlm_multilabel.data.images import copy_images_to_outdir, parse_image_size
from vlm_multilabel.data.conversion import build_record, write_jsonl


def parse_voc_xml(xml_path: str, include_difficult: bool = False) -> list[str]:
    """Parse a VOC XML annotation file, return list of object class names.

    By default difficult objects (<difficult>1</difficult>) are excluded,
    matching the standard PASCAL VOC 2007 multi-label ground-truth convention
    used by ML-GCN/ASL/CDUL (positive = non-difficult present). Pass
    include_difficult=True to count them as positive.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    names = []
    for obj in root.findall("object"):
        diff = obj.find("difficult")
        if not include_difficult and diff is not None and diff.text == "1":
            continue
        name = obj.find("name").text
        if name:
            names.append(name)
    return sorted(set(names))  # deduplicate, deterministic order


def to_record(
    config: VOCConfig,
    image_path: str,
    labels: list[str],
    sys_prompt: str,
) -> dict | None:
    """Build a VOC record, or return ``None`` when no positive label exists."""
    if not labels:
        return None
    return build_record(
        config,
        image_path,
        labels,
        sys_prompt,
        config.user_prompt,
    )

def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--voc-root", required=True,
                    help="VOCdevkit/VOC2007 or VOC2012 directory path")
    ap.add_argument("--out", default="data/voc", help="output directory")
    ap.add_argument("--splits", default="train:train,val:val",
                    help='comma list of "out_name:ImageSets_name" pairs '
                         '(ImageSets name without .txt); default '
                         'train:train,val:val (VOC2012); use '
                         'train:trainval,test:test for the VOC2007 '
                         'standard multi-label protocol')
    ap.add_argument("--include-difficult", action="store_true",
                    help="count difficult objects as positive labels; by "
                         "default they are excluded to match the standard "
                         "ML-GCN/ASL/CDUL GT (positive = non-difficult)")
    ap.add_argument("--primary-strategy", default="first", choices=["first"],
                    help="fixed first-positive primary selection in config order")
    ap.add_argument("--copy-images", action="store_true",
                    help="copy image files to <out>/images/ and rewrite paths")
    ap.add_argument("--image-size", default=None,
                    help="resize images to this size during copy, e.g. '448' "
                         "(square) or '448x448'; implies --copy-images")
    ap.add_argument("--no-pad", action="store_true",
                    help="force-resize without maintaining aspect ratio "
                         "(default: pad to square with black borders)")
    a = ap.parse_args(argv)

    config = VOCConfig()
    sys_prompt = config.render_prompt()

    voc_root = Path(a.voc_root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    image_size = parse_image_size(a.image_size) if a.image_size else None
    images_subdir = "images"
    if image_size is not None:
        if isinstance(image_size, int):
            images_subdir = f"images_{image_size}"
        else:
            images_subdir = f"images_{image_size[0]}x{image_size[1]}"

    # Split spec: each entry "out_name:ImageSets_name" (ImageSets name without
    # .txt). Default train:train,val:val mirrors VOC 2012. For the standard
    # multi-label VOC 2007 protocol pass --splits train:trainval,test:test
    # (train on the 5,011 trainval, evaluate on the 4,952 test).
    split_specs = []
    for spec in a.splits.split(","):
        out_name, _, imgset = spec.partition(":")
        out_name = out_name.strip()
        imgset = imgset.strip() or out_name
        split_specs.append((out_name, imgset))

    counts = {}
    for out_name, imgset in split_specs:
        set_file = voc_root / "ImageSets" / "Main" / f"{imgset}.txt"
        if set_file.exists():
            with open(set_file) as f:
                image_ids = [line.strip().split()[0]
                             for line in f if line.strip()]
        else:
            print(f"[warn] split file not found: {set_file}")
            image_ids = []

        records = []
        for img_id in image_ids:
            xml_path = voc_root / "Annotations" / f"{img_id}.xml"
            if not xml_path.exists():
                continue
            labels = parse_voc_xml(str(xml_path), a.include_difficult)
            if not labels:
                continue
            img_path = (voc_root / "JPEGImages" / f"{img_id}.jpg").as_posix()
            rec = to_record(config, img_path, labels, sys_prompt)
            if rec is None:
                continue
            records.append(rec)

        if a.copy_images or image_size is not None:
            copy_images_to_outdir(
                records, out,
                images_subdir=images_subdir,
                image_size=image_size,
                pad_to_square=not a.no_pad,
            )

        out_file = out / f"{out_name}.jsonl"
        write_jsonl(out_file, records)
        counts[out_name] = len(records)
        print(f"[{out_name} <- {imgset}.txt] {len(records):>5d} -> {out_file}")

    print(f"\ntotal={sum(counts.values())}  split={counts}")


if __name__ == "__main__":
    main()
