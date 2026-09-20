"""Shared utilities for data conversion scripts."""
from __future__ import annotations

import shutil
from pathlib import Path


def parse_image_size(s: str) -> int | tuple[int, int]:
    """Parse image size from string: '448' -> 448, '448x448' -> (448, 448)."""
    s = s.strip()
    if "x" in s.lower():
        w, h = s.lower().split("x", 1)
        return (int(w), int(h))
    return int(s)


def resize_image(
    src_path: Path,
    dst_path: Path,
    image_size: int | tuple[int, int],
    pad_to_square: bool = True,
) -> None:
    """Resize an image using LANCZOS interpolation and save to dst_path.

    When pad_to_square is True, maintains aspect ratio and pads the shorter
    dimension with black to reach the target size. Otherwise, force-resizes
    to the exact target dimensions. Preserves the original file format.
    """
    from PIL import Image

    if isinstance(image_size, int):
        target_w = target_h = image_size
    else:
        target_w, target_h = image_size

    img = Image.open(src_path).convert("RGB")

    if pad_to_square and img.size != (target_w, target_h):
        ratio = min(target_w / img.width, target_h / img.height)
        new_w = max(1, round(img.width * ratio))
        new_h = max(1, round(img.height * ratio))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        canvas.paste(img, ((target_w - new_w) // 2, (target_h - new_h) // 2))
        img = canvas
    else:
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "JPEG" if dst_path.suffix.lower() in (".jpg", ".jpeg") else "PNG"
    save_kwargs = {"quality": 95} if fmt == "JPEG" else {}
    img.save(dst_path, fmt, **save_kwargs)


def copy_images_to_outdir(
    records,
    out_dir,
    images_subdir="images",
    image_size=None,
    pad_to_square=True,
):
    """Copy or resize image files to <out_dir>/<images_subdir>/ and rewrite paths.

    Mutates records in place: updates each record's 'images' field to
    point to the copied/resized location. Uses forward-slash paths for
    cross-platform compatibility (matches existing jsonl convention).

    When image_size is set (int or (W, H) tuple), images are resized using
    LANCZOS interpolation. When pad_to_square is True (default), aspect
    ratio is preserved with black padding on the shorter dimension;
    otherwise images are force-resized to the exact target dimensions.

    Existing destination files are skipped (no overwrite), so re-running
    is safe and idempotent.
    """
    images_dir = Path(out_dir) / images_subdir
    images_dir.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    n_resized = 0
    n_skipped = 0
    n_missing = 0
    for rec in records:
        old_paths = rec.get("images", [])
        new_paths = []
        for old_path in old_paths:
            old_file = Path(old_path)
            if not old_file.exists():
                n_missing += 1
                new_paths.append(old_path)
                continue
            new_file = images_dir / old_file.name
            if not new_file.exists():
                if image_size is not None:
                    resize_image(old_file, new_file, image_size, pad_to_square)
                    n_resized += 1
                else:
                    shutil.copy2(old_path, new_file)
                    n_copied += 1
            else:
                n_skipped += 1
            new_paths.append(new_file.as_posix())
        rec["images"] = new_paths

    total = n_copied + n_resized + n_skipped + n_missing
    if image_size is not None:
        print(f"[copy-images] {n_resized} resized, {n_skipped} already present, "
              f"{n_missing} missing, {total} total -> {images_dir}")
    else:
        print(f"[copy-images] {n_copied} copied, {n_skipped} already present, "
              f"{n_missing} missing, {total} total -> {images_dir}")
