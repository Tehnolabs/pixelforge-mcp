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

# Supported image sizes (maps to google-genai ImageConfig.image_size).
# Per-model support varies; the Gemini API validates the combination.
IMAGE_SIZES = ["0.5K", "1K", "2K", "4K"]

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

DEFAULT_MODEL = "gemini-nano-banana-2.1"

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
# Vertex AI tools (optional, requires google-cloud-aiplatform)
# ---------------------------------------------------------------------------


class UpscaleImageInput(BaseModel):
    """Input validation for image upscaling (Vertex AI only)."""

    image_path: str = Field(..., description="Path to the image to upscale")
    upscale_factor: str = Field("x2", description="Upscale factor: 'x2' or 'x4'")
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
        if v not in ("x2", "x4"):
            raise ValueError("Upscale factor must be 'x2' or 'x4'")
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
    """Input validation for advanced editing (Vertex AI only)."""

    image_path: str = Field(..., description="Path to the image to edit")
    prompt: str = Field(..., description="Edit instruction")
    edit_mode: str = Field(
        ...,
        description="Edit mode: inpaint_removal, inpaint_insertion, "
        "outpaint, background_swap, style_transfer, product_image",
    )
    mask_path: Optional[str] = Field(
        None, description="Path to mask image (for inpainting)"
    )
    output_filename: Optional[str] = Field(None, description="Custom output filename")
    output_format: str = Field("png", description="Output format")

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, v: str) -> str:
        return _validate_image_path(v)

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
