"""Input validation for MCP tools."""

import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator

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

# Supported image sizes (maps to google-genai ImageConfig.image_size)
IMAGE_SIZES = ["1K", "2K", "4K"]

# Supported output formats
OUTPUT_FORMATS = ["png", "jpeg", "webp"]

# Format to file extension mapping
FORMAT_EXTENSIONS = {
    "png": ".png",
    "jpeg": ".jpg",
    "webp": ".webp",
}

# Person generation options (maps to google-genai ImageConfig.person_generation)
PERSON_GENERATION_OPTIONS = ["allow_all", "allow_adult", "dont_allow"]

# Maps user-friendly values to SDK values
PERSON_GENERATION_SDK_MAP = {
    "allow_all": "ALLOW_ALL",
    "allow_adult": "ALLOW_ADULT",
    "dont_allow": "ALLOW_NONE",
}


class GenerateImageInput(BaseModel):
    """Input validation for image generation."""

    prompt: str = Field(..., description="Text description of the image to generate")
    output_filename: Optional[str] = Field(
        None, description="Custom filename for the generated image"
    )
    aspect_ratio: str = Field(
        "1:1", description=f"Image aspect ratio. Options: {', '.join(ASPECT_RATIOS)}"
    )
    temperature: float = Field(
        0.7,
        description="Creativity level (0.0-2.0). Higher = more creative",
        ge=0.0,
        le=2.0,
    )
    model: Optional[str] = Field(
        None, description="Model to use (default: gemini-2.5-flash-image)"
    )
    safety_setting: str = Field(
        "preset:strict",
        description="Safety filter: preset:strict, preset:relaxed, or custom",
    )
    image_size: Optional[str] = Field(
        None,
        description="Output resolution: 1K (default), 2K, or 4K. "
        "4K requires gemini-3.1-flash or gemini-3-pro models.",
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
        description="Person generation control: allow_all, allow_adult, or dont_allow",
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate prompt is not empty and reasonable length."""
        v = v.strip()
        if not v:
            raise ValueError("Prompt cannot be empty")
        if len(v) > 2000:
            raise ValueError("Prompt too long (max 2000 characters)")
        return v

    @field_validator("aspect_ratio")
    @classmethod
    def validate_aspect_ratio(cls, v: str) -> str:
        """Validate aspect ratio is supported."""
        if v not in ASPECT_RATIOS:
            raise ValueError(
                f"Invalid aspect ratio. Must be one of: {', '.join(ASPECT_RATIOS)}"
            )
        return v

    @field_validator("output_filename")
    @classmethod
    def validate_filename(cls, v: Optional[str]) -> Optional[str]:
        """Validate filename is safe."""
        if v is None:
            return v

        # Check for path traversal attempts
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("Filename cannot contain path separators")

        # Check for valid characters
        if not re.match(r"^[a-zA-Z0-9_\-\.]+$", v):
            raise ValueError(
                "Filename can only contain letters, numbers, "
                "underscore, hyphen, and dot"
            )

        # Ensure it has an image extension
        if not v.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            v = v + ".png"

        return v

    @field_validator("image_size")
    @classmethod
    def validate_image_size(cls, v: Optional[str]) -> Optional[str]:
        """Validate image size is supported."""
        if v is None:
            return v
        v = v.upper()
        if v not in IMAGE_SIZES:
            raise ValueError(
                f"Invalid image size. Must be one of: {', '.join(IMAGE_SIZES)}"
            )
        return v

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        """Validate output format is supported."""
        v = v.lower()
        if v not in OUTPUT_FORMATS:
            raise ValueError(
                f"Invalid output format. Must be one of: {', '.join(OUTPUT_FORMATS)}"
            )
        return v

    @field_validator("person_generation")
    @classmethod
    def validate_person_generation(cls, v: Optional[str]) -> Optional[str]:
        """Validate person generation setting."""
        if v is None:
            return v
        v = v.lower()
        if v not in PERSON_GENERATION_OPTIONS:
            raise ValueError(
                f"Invalid person generation setting. "
                f"Must be one of: {', '.join(PERSON_GENERATION_OPTIONS)}"
            )
        return v


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
        "gemini-3-pro-image-preview is best for complex multi-turn edits.",
    )
    output_format: str = Field(
        "png",
        description="Output format: png (default), jpeg, or webp",
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate prompt."""
        v = v.strip()
        if not v:
            raise ValueError("Prompt cannot be empty")
        if len(v) > 2000:
            raise ValueError("Prompt too long (max 2000 characters)")
        return v

    @field_validator("input_image_path")
    @classmethod
    def validate_input_path(cls, v: str) -> str:
        """Validate input image exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Input image not found: {v}")
        if not path.is_file():
            raise ValueError(f"Input path is not a file: {v}")
        if path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".webp"]:
            raise ValueError(f"Invalid image format: {path.suffix}")
        return str(path.absolute())

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        """Validate output format is supported."""
        v = v.lower()
        if v not in OUTPUT_FORMATS:
            raise ValueError(
                f"Invalid output format. Must be one of: {', '.join(OUTPUT_FORMATS)}"
            )
        return v


class AnalyzeImageInput(BaseModel):
    """Input validation for image analysis."""

    image_path: str = Field(..., description="Path to the image to analyze")
    prompt: Optional[str] = Field(
        None,
        description="Custom analysis prompt (default: general description)",
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: Optional[str]) -> Optional[str]:
        """Validate optional analysis prompt."""
        if v is None:
            return v
        v = v.strip()
        if not v:
            raise ValueError("Prompt cannot be empty")
        if len(v) > 2000:
            raise ValueError("Prompt too long (max 2000 characters)")
        return v

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, v: str) -> str:
        """Validate image exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Image not found: {v}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {v}")
        if path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".webp"]:
            raise ValueError(f"Invalid image format: {path.suffix}")
        return str(path.absolute())
