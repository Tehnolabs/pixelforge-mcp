"""Unit tests for image transformation utilities."""

import pytest
from PIL import Image

from pixelforge_mcp.utils.transforms import (
    blur,
    crop,
    flip,
    grayscale,
    resize,
    rotate,
    sharpen,
    watermark,
)


class TestCrop:
    def test_crop_basic(self):
        img = Image.new("RGB", (100, 100), color="red")
        result = crop(img, 10, 10, 50, 50)
        assert result.size == (50, 50)

    def test_crop_full_image(self):
        img = Image.new("RGB", (100, 100), color="blue")
        result = crop(img, 0, 0, 100, 100)
        assert result.size == (100, 100)


class TestResize:
    def test_resize_exact(self):
        img = Image.new("RGB", (100, 100), color="red")
        result = resize(img, 50, 50, maintain_aspect=False)
        assert result.size == (50, 50)

    def test_resize_maintain_aspect(self):
        img = Image.new("RGB", (200, 100), color="red")
        result = resize(img, 100, 100, maintain_aspect=True)
        # thumbnail modifies in-place and fits within bounds
        assert result.size[0] <= 100
        assert result.size[1] <= 100


class TestRotate:
    def test_rotate_90(self):
        img = Image.new("RGB", (100, 50), color="red")
        result = rotate(img, 90)
        # Rotated 90 degrees with expand=True
        assert result.size[0] >= 50  # Width should be at least original height
        assert result.size[1] >= 100  # Height should be at least original width


class TestFlip:
    def test_flip_horizontal(self):
        img = Image.new("RGB", (100, 100), color="red")
        result = flip(img, "horizontal")
        assert result.size == (100, 100)

    def test_flip_vertical(self):
        img = Image.new("RGB", (100, 100), color="red")
        result = flip(img, "vertical")
        assert result.size == (100, 100)

    def test_flip_invalid_direction(self):
        img = Image.new("RGB", (100, 100))
        with pytest.raises(ValueError, match="Invalid direction"):
            flip(img, "diagonal")


class TestBlur:
    def test_blur_default(self):
        img = Image.new("RGB", (100, 100), color="red")
        result = blur(img)
        assert result.size == (100, 100)

    def test_blur_custom_radius(self):
        img = Image.new("RGB", (100, 100), color="red")
        result = blur(img, radius=5.0)
        assert result.size == (100, 100)


class TestSharpen:
    def test_sharpen_default(self):
        img = Image.new("RGB", (100, 100), color="red")
        result = sharpen(img)
        assert result.size == (100, 100)


class TestGrayscale:
    def test_grayscale_converts(self):
        img = Image.new("RGB", (100, 100), color="red")
        result = grayscale(img)
        assert result.size == (100, 100)
        assert result.mode == "RGB"  # Converted back to RGB


class TestWatermark:
    def test_watermark_default(self):
        img = Image.new("RGB", (200, 200), color="white")
        result = watermark(img, "Test Watermark")
        assert result.size == (200, 200)

    def test_watermark_center(self):
        img = Image.new("RGB", (200, 200), color="white")
        result = watermark(img, "Centered", position="center")
        assert result.size == (200, 200)

    def test_watermark_custom_opacity(self):
        img = Image.new("RGB", (200, 200), color="white")
        result = watermark(img, "Faded", opacity=0.1)
        assert result.size == (200, 200)
