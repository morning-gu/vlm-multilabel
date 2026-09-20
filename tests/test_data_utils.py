"""Tests for image resize and copy utilities."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from vlm_multilabel.data.images import (
    copy_images_to_outdir,
    parse_image_size,
    resize_image,
)


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Create a 200x150 RGB test image."""
    img = Image.new("RGB", (200, 150), (128, 64, 32))
    path = tmp_path / "test.png"
    img.save(path, "PNG")
    return path


class TestParseImageSize:
    def test_square_int_string(self):
        assert parse_image_size("448") == 448

    def test_wxh_string(self):
        assert parse_image_size("448x448") == (448, 448)

    def test_wxh_string_non_square(self):
        assert parse_image_size("224x336") == (224, 336)

    def test_strips_whitespace(self):
        assert parse_image_size("  448  ") == 448
        assert parse_image_size("224 x 336") == (224, 336)


class TestResizeImage:
    def test_square_resize_with_padding(self, sample_image, tmp_path):
        dst = tmp_path / "resized.png"
        resize_image(sample_image, dst, 128, pad_to_square=True)
        result = Image.open(dst)
        assert result.size == (128, 128)

    def test_force_resize_no_padding(self, sample_image, tmp_path):
        dst = tmp_path / "resized.png"
        resize_image(sample_image, dst, 128, pad_to_square=False)
        result = Image.open(dst)
        assert result.size == (128, 128)

    def test_non_square_target_with_padding(self, sample_image, tmp_path):
        dst = tmp_path / "resized.png"
        resize_image(sample_image, dst, (100, 200), pad_to_square=True)
        result = Image.open(dst)
        assert result.size == (100, 200)

    def test_preserves_png_format(self, sample_image, tmp_path):
        dst = tmp_path / "resized.png"
        resize_image(sample_image, dst, 64, pad_to_square=True)
        result = Image.open(dst)
        assert result.format == "PNG"

    def test_preserves_jpeg_format(self, sample_image, tmp_path):
        src = sample_image.with_suffix(".jpg")
        Image.open(sample_image).save(src, "JPEG")
        dst = tmp_path / "resized.jpg"
        resize_image(src, dst, 64, pad_to_square=True)
        result = Image.open(dst)
        assert result.format == "JPEG"


class TestCopyImagesToOutdir:
    @staticmethod
    def _make_record(image_path: Path) -> dict:
        return {
            "images": [str(image_path)],
            "label_vector": [0],
            "primary_code": "X",
            "messages": [],
        }

    def test_copy_without_resize(self, sample_image, tmp_path):
        records = [self._make_record(sample_image)]
        out_dir = tmp_path / "out"
        copy_images_to_outdir(records, out_dir)
        dst = out_dir / "images" / "test.png"
        assert dst.exists()

    def test_resize_with_int_size(self, sample_image, tmp_path):
        records = [self._make_record(sample_image)]
        out_dir = tmp_path / "out"
        copy_images_to_outdir(records, out_dir, image_size=64)
        dst = out_dir / "images" / "test.png"
        assert dst.exists()
        result = Image.open(dst)
        assert result.size == (64, 64)

    def test_resize_with_tuple_size(self, sample_image, tmp_path):
        records = [self._make_record(sample_image)]
        out_dir = tmp_path / "out"
        copy_images_to_outdir(records, out_dir, image_size=(48, 48))
        dst = out_dir / "images" / "test.png"
        assert dst.exists()
        result = Image.open(dst)
        assert result.size == (48, 48)

    def test_idempotent_skip(self, sample_image, tmp_path):
        records = [self._make_record(sample_image)]
        out_dir = tmp_path / "out"
        copy_images_to_outdir(records, out_dir, image_size=64)
        # Re-run should skip existing file without error
        records2 = [self._make_record(sample_image)]
        copy_images_to_outdir(records2, out_dir, image_size=64)

    def test_missing_source_keeps_original_path(self, tmp_path):
        records = [{"images": [str(tmp_path / "missing.png")]}]
        out_dir = tmp_path / "out"
        copy_images_to_outdir(records, out_dir, image_size=64)
        assert records[0]["images"][0] == str(tmp_path / "missing.png")
