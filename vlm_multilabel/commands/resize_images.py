"""Resize all images from one directory to another.

Scans a source directory for image files, resizes each to a uniform target
size using LANCZOS interpolation, and writes them to a destination directory
preserving the relative directory structure.

Usage:
  python -m scripts.resize_images \\
      --src /data/ChestX-ray14/images --dst /data/resized --image-size 448

  # Force-resize without aspect-ratio padding:
  python -m scripts.resize_images \\
      --src /data/raw --dst /data/out --image-size 448 --no-pad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from vlm_multilabel.data.images import parse_image_size, resize_image

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif", ".gif",
}


def collect_images(src_dir: Path, recursive: bool = True) -> list[Path]:
    """Collect all image file paths from a source directory."""
    if recursive:
        paths = [
            p for p in src_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    else:
        paths = [
            p for p in src_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    return sorted(paths)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--src", required=True, help="source image directory")
    ap.add_argument("--dst", required=True, help="destination directory")
    ap.add_argument(
        "--image-size", required=True,
        help="target size, e.g. '448' (square) or '448x448'",
    )
    ap.add_argument(
        "--no-pad", action="store_true",
        help="force-resize without maintaining aspect ratio "
             "(default: pad to square with black borders)",
    )
    ap.add_argument(
        "--no-recursive", action="store_true",
        help="only scan the top-level directory, not subdirectories",
    )
    a = ap.parse_args(argv)

    src_dir = Path(a.src)
    if not src_dir.is_dir():
        print(f"error: source directory not found: {src_dir}", file=sys.stderr)
        sys.exit(1)

    dst_dir = Path(a.dst)
    dst_dir.mkdir(parents=True, exist_ok=True)

    image_size = parse_image_size(a.image_size)
    pad_to_square = not a.no_pad
    recursive = not a.no_recursive

    images = collect_images(src_dir, recursive=recursive)
    if not images:
        print(f"no images found in {src_dir}")
        return

    print(f"found {len(images)} images in {src_dir}")
    print(f"resizing to {image_size} (pad={pad_to_square})")

    n_resized = 0
    n_skipped = 0
    n_failed = 0
    for src_path in tqdm(images, desc="resize", unit="img"):
        rel_path = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel_path
        if dst_path.exists():
            n_skipped += 1
            continue
        try:
            resize_image(src_path, dst_path, image_size, pad_to_square)
            n_resized += 1
        except Exception as exc:
            n_failed += 1
            print(f"  failed: {src_path} -> {exc}", file=sys.stderr)

    print(f"\ndone: {n_resized} resized, {n_skipped} skipped, {n_failed} failed")


if __name__ == "__main__":
    main()
