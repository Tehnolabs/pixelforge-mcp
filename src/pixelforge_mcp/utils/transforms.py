"""Image transformation utilities using Pillow."""

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


def crop(image: Image.Image, x: int, y: int, width: int, height: int) -> Image.Image:
    """Crop image to specified region."""
    return image.crop((x, y, x + width, y + height))


def resize(
    image: Image.Image,
    width: int,
    height: int,
    maintain_aspect: bool = True,
) -> Image.Image:
    """Resize image, optionally maintaining aspect ratio."""
    if maintain_aspect:
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        return image
    return image.resize((width, height), Image.Resampling.LANCZOS)


def rotate(image: Image.Image, degrees: float) -> Image.Image:
    """Rotate image by degrees (counterclockwise)."""
    return image.rotate(degrees, expand=True, resample=Image.Resampling.BICUBIC)


def flip(image: Image.Image, direction: str) -> Image.Image:
    """Flip image horizontally or vertically."""
    if direction == "horizontal":
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    elif direction == "vertical":
        return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    raise ValueError(f"Invalid direction: {direction}. Use 'horizontal' or 'vertical'.")


def blur(image: Image.Image, radius: float = 2.0) -> Image.Image:
    """Apply Gaussian blur."""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def sharpen(image: Image.Image, factor: float = 2.0) -> Image.Image:
    """Sharpen image. Factor > 1.0 increases sharpness."""
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)


def grayscale(image: Image.Image) -> Image.Image:
    """Convert image to grayscale."""
    return image.convert("L").convert("RGB")


def watermark(
    image: Image.Image,
    text: str,
    position: str = "bottom-right",
    opacity: float = 0.5,
    font_size: int = 24,
) -> Image.Image:
    """Add text watermark to image."""
    # Work on a copy with alpha channel
    watermarked = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", watermarked.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Try to use a default font
    try:
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont = ImageFont.truetype(
            "Arial", font_size
        )
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Calculate position
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    padding = 10
    positions = {
        "top-left": (padding, padding),
        "top-right": (watermarked.width - text_width - padding, padding),
        "bottom-left": (
            padding,
            watermarked.height - text_height - padding,
        ),
        "bottom-right": (
            watermarked.width - text_width - padding,
            watermarked.height - text_height - padding,
        ),
        "center": (
            (watermarked.width - text_width) // 2,
            (watermarked.height - text_height) // 2,
        ),
    }
    pos = positions.get(position, positions["bottom-right"])

    alpha = int(255 * opacity)
    draw.text(pos, text, fill=(255, 255, 255, alpha), font=font)

    watermarked = Image.alpha_composite(watermarked, overlay)
    return watermarked.convert(image.mode)
