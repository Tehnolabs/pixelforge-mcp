"""Input validation for MCP tools."""

import re
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field, field_validator, model_validator

# Supported aspect ratios
ASPECT_RATIOS = [
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
    "1:4",
    "4:1",
    "1:8",
    "8:1",
]

# Ratios wider or taller than 21:9. Only some models honour them.
PANORAMIC_RATIOS = ["1:4", "4:1", "1:8", "8:1"]

# Models that honour the panoramic ratios (Gemini API docs, 2026-10). Other
# known models render a standard shape instead of failing.
PANORAMIC_MODELS = {"gemini-nano-banana-2.1", "gemini-3.1-flash-image"}

# Supported image sizes as exposed by the tools.
IMAGE_SIZES = ["0.5K", "1K", "2K", "4K"]

# Values the API expects for ImageConfig.image_size. Google documents the
# smallest tier as "512px"; "0.5K" is rejected by Vertex AI.
SDK_IMAGE_SIZE_VALUES: Dict[str, str] = {
    "0.5K": "512px",
    "1K": "1K",
    "2K": "2K",
    "4K": "4K",
}

# Longest output edge in pixels per size tier.
IMAGE_SIZE_PIXELS: Dict[str, int] = {
    "0.5K": 512,
    "1K": 1024,
    "2K": 2048,
    "4K": 4096,
}

# Sizes each model can output (Google model cards, 2026-10).
MODEL_IMAGE_SIZES: Dict[str, List[str]] = {
    "gemini-nano-banana-2.1": ["1K", "2K", "4K"],
    "gemini-3.1-flash-lite-image": ["1K"],
    "gemini-3.1-flash-image": ["0.5K", "1K", "2K", "4K"],
    "gemini-3-pro-image": ["1K", "2K", "4K"],
    "gemini-2.5-flash-image": ["1K"],
}


def supported_image_sizes(model: str) -> Optional[List[str]]:
    """Return the sizes a known model supports, or None for unknown models."""
    return MODEL_IMAGE_SIZES.get(model)


def check_model_image_size(model: Optional[str], image_size: Optional[str]) -> None:
    """Raise ValueError when a known model cannot output the requested size."""
    if not model or not image_size:
        return
    sizes = supported_image_sizes(model)
    if sizes is not None and image_size not in sizes:
        raise ValueError(
            f"{model} cannot output {image_size}. "
            f"Supported sizes: {', '.join(sizes)}"
        )


def supported_aspect_ratios(model: Optional[str]) -> List[str]:
    """Return the ratios a model honours; unknown models get the full list."""
    if model is None or model not in IMAGE_MODELS or model in PANORAMIC_MODELS:
        return list(ASPECT_RATIOS)
    return [ratio for ratio in ASPECT_RATIOS if ratio not in PANORAMIC_RATIOS]


# Supported output formats
OUTPUT_FORMATS = ["png", "jpeg", "webp"]

# Format to file extension mapping
FORMAT_EXTENSIONS = {
    "png": ".png",
    "jpeg": ".jpg",
    "webp": ".webp",
}

# Person generation values are accepted for compatibility but are not
# enforced by the Gemini API, which applies its own built-in restrictions.
PERSON_GENERATION_OPTIONS = ["allow", "adults_only", "block"]

# Prompt optimization style presets
PROMPT_STYLES = [
    "photorealistic",
    "illustration",
    "3d_render",
    "pixel_art",
    "watercolor",
    "oil_painting",
    "sketch",
    "anime",
    "cinematic",
    "product_photo",
    "architecture",
    "food",
    "fashion",
    "abstract",
]

# Stable Gemini image models (all served via generate_content)
IMAGE_MODELS: List[str] = [
    "gemini-nano-banana-2.1",
    "gemini-3.1-flash-lite-image",
    "gemini-3.1-flash-image",
    "gemini-3-pro-image",
    "gemini-2.5-flash-image",  # deprecated legacy
]

# Text models used by the analysis tools
ANALYSIS_MODELS: List[str] = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
]

DEFAULT_MODEL = "gemini-nano-banana-2.1"
DEFAULT_ANALYSIS_MODEL = "gemini-3.5-flash-lite"

# Preview IDs Google shut down on 2026-06-25, mapped to the stable
# replacements named on the Gemini API deprecations page.
DEPRECATED_MODEL_ALIASES: Dict[str, str] = {
    "gemini-3-pro-image-preview": "gemini-3-pro-image",
    "gemini-3.1-flash-image-preview": "gemini-nano-banana-2.1",
}

# Imagen 4 models Google shut down on 2026-08-17 (the API returns 404).
RETIRED_MODELS: Dict[str, str] = {
    "imagen-4.0-generate-001": "gemini-nano-banana-2.1",
    "imagen-4.0-fast-generate-001": "gemini-nano-banana-2.1",
    "imagen-4.0-ultra-generate-001": "gemini-nano-banana-2.1",
}

# Cost per operation per model (USD).
# Image models: price per image at 1K for generate/edit.
# Analysis models: per-call estimate for a single image plus a short prompt.
COST_TABLE = {
    "gemini-nano-banana-2.1": {
        "generate": 0.0336,
        "edit": 0.0336,
    },
    "gemini-3.1-flash-lite-image": {
        "generate": 0.0336,
        "edit": 0.0336,
    },
    "gemini-3.1-flash-image": {
        "generate": 0.067,
        "edit": 0.067,
    },
    "gemini-3-pro-image": {
        "generate": 0.134,
        "edit": 0.134,
    },
    "gemini-2.5-flash-image": {
        "generate": 0.039,
        "edit": 0.039,
    },
    "gemini-3.5-flash-lite": {
        "analyze": 0.001,
    },
    "gemini-3.8-flash": {
        "analyze": 0.004,
    },
}


class QualityPreset(TypedDict):
    """Model and image size pair behind a quality preset name."""

    model: str
    image_size: Optional[str]


# Quality presets map to model + image_size combinations
QUALITY_PRESETS: Dict[str, QualityPreset] = {
    "fast": {
        "model": "gemini-3.1-flash-lite-image",
        "image_size": None,
    },
    "balanced": {
        "model": "gemini-nano-banana-2.1",
        "image_size": "1K",
    },
    "quality": {
        "model": "gemini-3-pro-image",
        "image_size": "2K",
    },
}

# Valid image file extensions
VALID_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]


def _validate_image_path(v: str) -> str:
    """Validate an image path exists and has a valid format."""
    path = Path(v)
    if not path.exists():
        raise ValueError(f"Image not found: {v}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {v}")
    if path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
        raise ValueError(f"Invalid image format: {path.suffix}")
    return str(path.absolute())


def _validate_prompt_text(v: str) -> str:
    """Validate a prompt string (non-empty, max 2000 chars)."""
    v = v.strip()
    if not v:
        raise ValueError("Prompt cannot be empty")
    if len(v) > 2000:
        raise ValueError("Prompt too long (max 2000 characters)")
    return v


def _validate_optional_prompt(v: Optional[str]) -> Optional[str]:
    """Validate an optional prompt string."""
    if v is None:
        return v
    return _validate_prompt_text(v)


def _validate_output_filename(v: Optional[str]) -> Optional[str]:
    """Validate an output filename (no path traversal, safe characters only)."""
    if v is None:
        return v
    if ".." in v or "/" in v or "\\" in v:
        raise ValueError("Filename cannot contain path separators")
    if not re.match(r"^[a-zA-Z0-9_\-\.]+$", v):
        raise ValueError(
            "Filename can only contain letters, numbers, " "underscore, hyphen, and dot"
        )
    if not v.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        v = v + ".png"
    return v


def _validate_output_format(v: str) -> str:
    """Validate output format (png, jpeg, webp)."""
    v = v.lower()
    if v not in OUTPUT_FORMATS:
        raise ValueError(
            f"Invalid output format. " f"Must be one of: {', '.join(OUTPUT_FORMATS)}"
        )
    return v


# ---------------------------------------------------------------------------
# Generation tools
# ---------------------------------------------------------------------------


class GenerateImageInput(BaseModel):
    """Input validation for image generation."""

    prompt: str = Field(..., description="Text description of the image to generate")
    output_filename: Optional[str] = Field(
        None, description="Custom filename for the generated image"
    )
    aspect_ratio: str = Field(
        "1:1",
        description=(f"Image aspect ratio. Options: {', '.join(ASPECT_RATIOS)}"),
    )
    temperature: float = Field(
        0.7,
        description="Creativity level (0.0-2.0). Higher = more creative",
        ge=0.0,
        le=2.0,
    )
    model: Optional[str] = Field(
        None, description=f"Model to use (default: {DEFAULT_MODEL})"
    )
    quality: Optional[str] = Field(
        None,
        description="Quality preset: 'fast' (cheapest, fastest), "
        "'balanced' (good quality, fast), or 'quality' (best output). "
        "Mutually exclusive with 'model'.",
    )
    safety_setting: str = Field(
        "preset:strict",
        description="Safety filter: preset:strict, preset:relaxed",
    )
    image_size: Optional[str] = Field(
        None,
        description="Output resolution: 0.5K (gemini-3.1-flash-image only), "
        "1K (default), 2K, or 4K. 2K/4K are not supported by "
        "gemini-3.1-flash-lite-image.",
    )
    number_of_images: int = Field(
        1,
        description="Number of image variations to generate (1-4)",
        ge=1,
        le=4,
    )
    output_format: str = Field(
        "png",
        description="Output format: png (default), jpeg, or webp",
    )
    person_generation: Optional[str] = Field(
        None,
        description="Accepted for compatibility. Not enforced by the "
        "Gemini API; Gemini applies its own built-in restrictions on "
        "identifiable people.",
    )
    reference_images: Optional[List[str]] = Field(
        None,
        description="Paths to reference images for style/character "
        "consistency. Up to 14 images on gemini-nano-banana-2.1 and "
        "gemini-3.1-flash-image; 6 on gemini-3-pro-image.",
    )
    thinking_budget: Optional[int] = Field(
        None,
        description="Thinking budget for extended reasoning (0-24576 tokens). "
        "Higher values = deeper reasoning. Only for Gemini models.",
        ge=0,
        le=24576,
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        return _validate_prompt_text(v)

    @field_validator("aspect_ratio")
    @classmethod
    def validate_aspect_ratio(cls, v: str) -> str:
        if v not in ASPECT_RATIOS:
            raise ValueError(
                f"Invalid aspect ratio. " f"Must be one of: {', '.join(ASPECT_RATIOS)}"
            )
        return v

    @field_validator("output_filename")
    @classmethod
    def validate_filename(cls, v: Optional[str]) -> Optional[str]:
        return _validate_output_filename(v)

    @field_validator("image_size")
    @classmethod
    def validate_image_size(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.upper()
        if v not in IMAGE_SIZES:
            raise ValueError(
                f"Invalid image size. " f"Must be one of: {', '.join(IMAGE_SIZES)}"
            )
        return v

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        return _validate_output_format(v)

    @field_validator("person_generation")
    @classmethod
    def validate_person_generation(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.lower()
        if v not in PERSON_GENERATION_OPTIONS:
            raise ValueError(
                f"Invalid person generation setting. "
                f"Must be one of: {', '.join(PERSON_GENERATION_OPTIONS)}"
            )
        return v

    @field_validator("reference_images")
    @classmethod
    def validate_reference_images(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return v
        if len(v) > 14:
            raise ValueError("Maximum 14 reference images allowed")
        if len(v) == 0:
            return None
        return [_validate_image_path(p) for p in v]

    @field_validator("quality")
    @classmethod
    def validate_quality(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.lower()
        if v not in QUALITY_PRESETS:
            raise ValueError(
                f"Invalid quality preset. "
                f"Must be one of: {', '.join(QUALITY_PRESETS)}"
            )
        return v

    @model_validator(mode="after")
    def check_quality_model_exclusive(self) -> "GenerateImageInput":
        if self.quality and self.model:
            raise ValueError(
                "'quality' and 'model' are mutually " "exclusive. Use one or the other."
            )
        # Check the model and size the server will send: a preset fixes the
        # model and may leave the caller's image_size in place.
        model = self.model
        image_size = self.image_size
        if self.quality:
            preset = QUALITY_PRESETS[self.quality]
            model = preset["model"]
            image_size = preset["image_size"] or self.image_size
        check_model_image_size(model, image_size)
        return self


class EditImageInput(BaseModel):
    """Input validation for image editing."""

    prompt: str = Field(..., description="Description of desired changes")
    input_image_path: str = Field(..., description="Path to the image to edit")
    output_filename: Optional[str] = Field(
        None, description="Custom filename for the edited image"
    )
    temperature: float = Field(
        0.7, description="Creativity level (0.0-2.0)", ge=0.0, le=2.0
    )
    model: Optional[str] = Field(
        None,
        description="Model to use for editing. "
        "gemini-nano-banana-2.1 (default) or gemini-3-pro-image "
        "for complex multi-turn edits.",
    )
    output_format: str = Field(
        "png",
        description="Output format: png (default), jpeg, or webp",
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        return _validate_prompt_text(v)

    @field_validator("input_image_path")
    @classmethod
    def validate_input_path(cls, v: str) -> str:
        return _validate_image_path(v)

    @field_validator("output_filename")
    @classmethod
    def validate_filename(cls, v: Optional[str]) -> Optional[str]:
        return _validate_output_filename(v)

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        return _validate_output_format(v)


# ---------------------------------------------------------------------------
# Analysis tools
# ---------------------------------------------------------------------------


class AnalyzeImageInput(BaseModel):
    """Input validation for image analysis."""

    image_path: str = Field(..., description="Path to the image to analyze")
    prompt: Optional[str] = Field(
        None,
        description="Custom analysis prompt (default: general description)",
    )
    model: Optional[str] = Field(
        None,
        description="Analysis model override. "
        f"Default {DEFAULT_ANALYSIS_MODEL}. "
        "Use gemini-3.8-flash for harder OCR or reasoning.",
    )
    use_grounding: bool = Field(
        False,
        description="Enable Google Search grounding for more accurate, "
        "factual analysis results",
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: Optional[str]) -> Optional[str]:
        return _validate_optional_prompt(v)

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, v: str) -> str:
        return _validate_image_path(v)


class ExtractTextInput(BaseModel):
    """Input validation for OCR / text extraction."""

    image_path: str = Field(..., description="Path to the image to extract text from")
    model: Optional[str] = Field(
        None,
        description="Analysis model override. "
        f"Default {DEFAULT_ANALYSIS_MODEL}. "
        "Use gemini-3.8-flash for harder OCR or reasoning.",
    )
    use_grounding: bool = Field(
        False,
        description="Enable Google Search grounding for more accurate, "
        "factual analysis results",
    )

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, v: str) -> str:
        return _validate_image_path(v)


class DetectObjectsInput(BaseModel):
    """Input validation for object detection."""

    image_path: str = Field(..., description="Path to the image to detect objects in")
    objects: Optional[str] = Field(
        None,
        description="Specific objects to detect (e.g. 'cats and dogs'). "
        "If not provided, detects all visible objects.",
    )
    model: Optional[str] = Field(
        None,
        description="Analysis model override. "
        f"Default {DEFAULT_ANALYSIS_MODEL}. "
        "Use gemini-3.8-flash for harder OCR or reasoning.",
    )
    use_grounding: bool = Field(
        False,
        description="Enable Google Search grounding for more accurate, "
        "factual analysis results",
    )

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, v: str) -> str:
        return _validate_image_path(v)


class CompareImagesInput(BaseModel):
    """Input validation for image comparison."""

    image_paths: List[str] = Field(
        ..., description="Paths to 2 or more images to compare"
    )
    prompt: Optional[str] = Field(
        None,
        description="Comparison focus (e.g. 'color differences', "
        "'layout changes'). Default: general comparison.",
    )
    model: Optional[str] = Field(
        None,
        description="Analysis model override. "
        f"Default {DEFAULT_ANALYSIS_MODEL}. "
        "Use gemini-3.8-flash for harder OCR or reasoning.",
    )
    use_grounding: bool = Field(
        False,
        description="Enable Google Search grounding for more accurate, "
        "factual analysis results",
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: Optional[str]) -> Optional[str]:
        return _validate_optional_prompt(v)

    @field_validator("image_paths")
    @classmethod
    def validate_image_paths(cls, v: List[str]) -> List[str]:
        if len(v) < 2:
            raise ValueError("At least 2 images required for comparison")
        if len(v) > 10:
            raise ValueError("Maximum 10 images for comparison")
        return [_validate_image_path(p) for p in v]


class RemoveBackgroundInput(BaseModel):
    """Input validation for background removal."""

    image_path: str = Field(
        ..., description="Path to the image to remove background from"
    )
    output_filename: Optional[str] = Field(
        None, description="Custom filename for the result"
    )
    output_format: str = Field(
        "png",
        description="Output format: png (default, supports transparency), "
        "jpeg, or webp",
    )

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, v: str) -> str:
        return _validate_image_path(v)

    @field_validator("output_filename")
    @classmethod
    def validate_filename(cls, v: Optional[str]) -> Optional[str]:
        return _validate_output_filename(v)

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        return _validate_output_format(v)


# Supported image transform operations
TRANSFORM_OPERATIONS = [
    "crop",
    "resize",
    "rotate",
    "flip",
    "blur",
    "sharpen",
    "grayscale",
    "watermark",
]


class TransformImageInput(BaseModel):
    """Input validation for image transformations."""

    image_path: str = Field(..., description="Path to the image to transform")
    operation: str = Field(
        ...,
        description="Transform operation: crop, resize, rotate, flip, "
        "blur, sharpen, grayscale, or watermark",
    )
    output_filename: Optional[str] = Field(
        None, description="Custom filename for the result"
    )
    output_format: str = Field("png", description="Output format: png, jpeg, or webp")

    # Operation-specific parameters
    x: Optional[int] = Field(None, description="Crop: x offset")
    y: Optional[int] = Field(None, description="Crop: y offset")
    width: Optional[int] = Field(None, description="Crop/resize: width in pixels")
    height: Optional[int] = Field(None, description="Crop/resize: height in pixels")
    maintain_aspect: bool = Field(True, description="Resize: maintain aspect ratio")
    degrees: Optional[float] = Field(None, description="Rotate: degrees")
    direction: Optional[str] = Field(
        None, description="Flip: 'horizontal' or 'vertical'"
    )
    radius: float = Field(2.0, description="Blur: radius")
    factor: float = Field(2.0, description="Sharpen: factor (>1 = sharper)")
    text: Optional[str] = Field(None, description="Watermark: text to overlay")
    position: str = Field(
        "bottom-right",
        description="Watermark position: top-left, top-right, "
        "bottom-left, bottom-right, center",
    )
    opacity: float = Field(0.5, description="Watermark opacity (0.0-1.0)")

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, v: str) -> str:
        return _validate_image_path(v)

    @field_validator("output_filename")
    @classmethod
    def validate_filename(cls, v: Optional[str]) -> Optional[str]:
        return _validate_output_filename(v)

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        return _validate_output_format(v)

    @field_validator("operation")
    @classmethod
    def validate_operation(cls, v: str) -> str:
        v = v.lower()
        if v not in TRANSFORM_OPERATIONS:
            raise ValueError(
                f"Invalid operation. "
                f"Must be one of: {', '.join(TRANSFORM_OPERATIONS)}"
            )
        return v


# ---------------------------------------------------------------------------
# Utility tools
# ---------------------------------------------------------------------------


class OptimizePromptInput(BaseModel):
    """Input validation for prompt optimization."""

    prompt: str = Field(..., description="Basic prompt to enhance for better results")
    style: Optional[str] = Field(
        None,
        description="Target style: photorealistic, illustration, "
        "3d_render, pixel_art, watercolor, oil_painting, sketch, "
        "anime, cinematic, product_photo, architecture, food, "
        "fashion, abstract",
    )
    model: Optional[str] = Field(
        None,
        description="Analysis model override. "
        f"Default {DEFAULT_ANALYSIS_MODEL}. "
        "Use gemini-3.8-flash for harder OCR or reasoning.",
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        return _validate_prompt_text(v)

    @field_validator("style")
    @classmethod
    def validate_style(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.lower()
        if v not in PROMPT_STYLES:
            raise ValueError(
                f"Invalid style. " f"Must be one of: {', '.join(PROMPT_STYLES)}"
            )
        return v


class EstimateCostInput(BaseModel):
    """Input validation for cost estimation."""

    operation: str = Field(
        ...,
        description="Operation type: generate, edit, or analyze",
    )
    model: Optional[str] = Field(
        None,
        description=f"Model to estimate cost for. Default: {DEFAULT_MODEL}",
    )
    number_of_images: int = Field(1, description="Number of images", ge=1, le=100)

    @field_validator("operation")
    @classmethod
    def validate_operation(cls, v: str) -> str:
        v = v.lower()
        if v not in ["generate", "edit", "analyze"]:
            raise ValueError(
                "Invalid operation. " "Must be one of: generate, edit, analyze"
            )
        return v


# ---------------------------------------------------------------------------
# Template tools
# ---------------------------------------------------------------------------


class ListTemplatesInput(BaseModel):
    """Input validation for listing templates."""

    category: Optional[str] = Field(
        None,
        description="Filter by category: product_photography, social_media, "
        "illustration, portrait, architecture, food, fashion, "
        "abstract, logo, panoramic",
    )


class ApplyTemplateInput(BaseModel):
    """Input validation for applying a template."""

    template_name: str = Field(..., description="Template name (e.g., 'product_hero')")
    subject: str = Field(..., description="The subject to fill into the template")

    @field_validator("template_name")
    @classmethod
    def validate_template_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Template name cannot be empty")
        return v

    @field_validator("subject")
    @classmethod
    def validate_subject(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Subject cannot be empty")
        if len(v) > 500:
            raise ValueError("Subject too long (max 500 characters)")
        return v


# ---------------------------------------------------------------------------
# Generative upscale and mode-driven editing
# ---------------------------------------------------------------------------


UPSCALE_FACTORS: Dict[str, int] = {"x2": 2, "x4": 4}


def _validate_generative_model(v: Optional[str]) -> Optional[str]:
    """Normalise an optional model override (blank means default)."""
    if v is None:
        return v
    v = v.strip()
    if not v:
        return None
    return v


class UpscaleImageInput(BaseModel):
    """Input validation for generative image upscaling."""

    image_path: str = Field(..., description="Path to the image to upscale")
    upscale_factor: str = Field("x2", description="Upscale factor: 'x2' or 'x4'")
    model: Optional[str] = Field(
        None,
        description=f"Image model to re-render with (default: {DEFAULT_MODEL}). "
        "Must support 2K/4K output.",
    )
    output_filename: Optional[str] = Field(None, description="Custom output filename")
    output_format: str = Field("png", description="Output format")

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, v: str) -> str:
        return _validate_image_path(v)

    @field_validator("upscale_factor")
    @classmethod
    def validate_upscale_factor(cls, v: str) -> str:
        v = v.lower()
        if v not in UPSCALE_FACTORS:
            raise ValueError("Upscale factor must be 'x2' or 'x4'")
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: Optional[str]) -> Optional[str]:
        v = _validate_generative_model(v)
        if v is None:
            return v
        sizes = supported_image_sizes(v)
        if sizes is not None and sizes == ["1K"]:
            raise ValueError(
                f"{v} outputs 1K only and cannot upscale. "
                f"Use {DEFAULT_MODEL} or gemini-3-pro-image."
            )
        return v

    @field_validator("output_filename")
    @classmethod
    def validate_filename(cls, v: Optional[str]) -> Optional[str]:
        return _validate_output_filename(v)

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        return _validate_output_format(v)


ADVANCED_EDIT_MODES = [
    "inpaint_removal",
    "inpaint_insertion",
    "outpaint",
    "background_swap",
    "style_transfer",
    "product_image",
]


class AdvancedEditInput(BaseModel):
    """Input validation for mode-driven editing with a Gemini image model."""

    image_path: str = Field(..., description="Path to the image to edit")
    prompt: str = Field(..., description="Edit instruction")
    edit_mode: str = Field(
        ...,
        description="Edit mode: inpaint_removal, inpaint_insertion, "
        "outpaint, background_swap, style_transfer, product_image",
    )
    mask_path: Optional[str] = Field(
        None,
        description="Optional mask image. White marks the region to change. "
        "It guides the model; the boundary is not pixel-exact.",
    )
    model: Optional[str] = Field(
        None, description=f"Image model to use (default: {DEFAULT_MODEL})"
    )
    output_filename: Optional[str] = Field(None, description="Custom output filename")
    output_format: str = Field("png", description="Output format")

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, v: str) -> str:
        return _validate_image_path(v)

    @field_validator("mask_path")
    @classmethod
    def validate_mask_path(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return _validate_image_path(v)

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: Optional[str]) -> Optional[str]:
        return _validate_generative_model(v)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        return _validate_prompt_text(v)

    @field_validator("edit_mode")
    @classmethod
    def validate_edit_mode(cls, v: str) -> str:
        v = v.lower()
        if v not in ADVANCED_EDIT_MODES:
            raise ValueError(
                f"Invalid edit mode. "
                f"Must be one of: {', '.join(ADVANCED_EDIT_MODES)}"
            )
        return v

    @field_validator("output_filename")
    @classmethod
    def validate_filename(cls, v: Optional[str]) -> Optional[str]:
        return _validate_output_filename(v)

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        return _validate_output_format(v)
