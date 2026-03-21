"""Main MCP server implementation for Gemini Imagen."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastmcp import FastMCP
from pydantic import ValidationError

from .config import get_config
from .utils.api_client import GenerationResult, ImagenAPIClient
from .utils.validation import (
    COST_TABLE,
    FORMAT_EXTENSIONS,
    QUALITY_PRESETS,
    AnalyzeImageInput,
    ApplyTemplateInput,
    CompareImagesInput,
    DetectObjectsInput,
    EditImageInput,
    EstimateCostInput,
    ExtractTextInput,
    GenerateImageInput,
    ListTemplatesInput,
    OptimizePromptInput,
    RemoveBackgroundInput,
    TransformImageInput,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Initialize FastMCP server
mcp = FastMCP("gemini-imagen-mcp")


# Initialize API client (will be set on first use)
_client: Optional[ImagenAPIClient] = None


def get_api_client() -> ImagenAPIClient:
    """Get or create API client instance."""
    global _client
    if _client is None:
        config = get_config()
        api_key = (
            config.imagen.api_key.get_secret_value() if config.imagen.api_key else None
        )
        _client = ImagenAPIClient(
            model_name=config.imagen.default_model,
            api_key=api_key,
            log_images=False,
        )
    return _client


def generate_output_path(
    filename: Optional[str] = None,
    prefix: str = "generated",
    output_format: str = "png",
) -> Path:
    """Generate a unique output path for an image."""
    config = get_config()
    output_dir = config.storage.output_dir

    if filename:
        return output_dir / filename

    ext = FORMAT_EXTENSIONS.get(output_format, ".png")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return output_dir / f"{prefix}_{timestamp}{ext}"


def format_image_result(
    result: GenerationResult,
    output_paths: list[Path],
) -> dict:
    """Format image generation/edit result for MCP response."""
    if not result.success:
        return {
            "success": False,
            "message": result.error or "Operation failed",
        }

    images = []
    for p in output_paths:
        if p.exists():
            images.append(
                {
                    "path": str(p.absolute()),
                    "size_bytes": p.stat().st_size,
                }
            )

    response: dict = {
        "success": True,
        "message": result.output,
        "images": images,
    }
    if result.data:
        response["details"] = result.data
    return response


# ------------------------------------------------------------------
# Generation tools
# ------------------------------------------------------------------


@mcp.tool()
async def generate_image(
    prompt: str,
    output_filename: Optional[str] = None,
    aspect_ratio: str = "1:1",
    temperature: float = 0.7,
    model: Optional[str] = None,
    quality: Optional[str] = None,
    safety_setting: str = "preset:strict",
    image_size: Optional[str] = None,
    number_of_images: int = 1,
    output_format: str = "png",
    person_generation: Optional[str] = None,
    reference_images: Optional[List[str]] = None,
) -> dict:
    """Generate an image from a text prompt using Google Gemini.

    Args:
        prompt: Text description of the image to generate
        output_filename: Custom filename (optional, auto-generated
            if not provided)
        aspect_ratio: Image dimensions (1:1, 16:9, 9:16, etc.)
        temperature: Creativity level 0.0-2.0
        model: Model override. Options:
            - "gemini-2.5-flash-image" (default): Fast, cheap
            - "gemini-3-pro-image-preview": Best text, complex edits
            - "gemini-3.1-flash-image-preview": Panoramic, fast 4K
        quality: Quality preset: "fast" (cheapest), "balanced"
            (good quality), or "quality" (best output). Mutually
            exclusive with 'model'.
        safety_setting: Safety filter (preset:strict, preset:relaxed)
        image_size: Resolution: "1K" (default), "2K", or "4K"
        number_of_images: Generate 1-4 variations (default: 1)
        output_format: File format: "png", "jpeg", or "webp"
        person_generation: People in images: "allow", "adults_only"
            (no minors), or "block" (no people)
        reference_images: Paths to reference images for style/character
            consistency (up to 14 images)

    Returns:
        Dictionary with generation result and image path(s)

    Example:
        generate_image(prompt="A sunset over mountains")

        generate_image(
            prompt="Product photo of a watch",
            quality="quality"
        )

        generate_image(
            prompt="Product photo of a watch",
            model="gemini-3.1-flash-image-preview",
            image_size="4K",
            output_format="webp"
        )
    """
    logger.info(f"Generating image: {prompt[:50]}...")

    try:
        inputs = GenerateImageInput(
            prompt=prompt,
            output_filename=output_filename,
            aspect_ratio=aspect_ratio,
            temperature=temperature,
            model=model,
            quality=quality,
            safety_setting=safety_setting,
            image_size=image_size,
            number_of_images=number_of_images,
            output_format=output_format,
            person_generation=person_generation,
            reference_images=reference_images,
        )

        # Resolve quality preset
        if inputs.quality:
            preset = QUALITY_PRESETS[inputs.quality]
            resolved_model = preset["model"]
            resolved_image_size = preset["image_size"] or inputs.image_size
        else:
            resolved_model = inputs.model
            resolved_image_size = inputs.image_size

        client = get_api_client()
        n = inputs.number_of_images
        paths = []

        # Build all paths first
        for i in range(n):
            suffix = f"_{i + 1}" if n > 1 else ""
            if inputs.output_filename and n > 1:
                p = Path(inputs.output_filename)
                name = f"{p.stem}{suffix}{p.suffix or '.png'}"
            elif inputs.output_filename:
                name = inputs.output_filename
            else:
                name = None

            path = generate_output_path(
                filename=name,
                prefix=f"generated{suffix}",
                output_format=inputs.output_format,
            )
            paths.append(path)

        # Generate all images in parallel
        async def _generate_one(path: Path) -> GenerationResult:
            return await client.generate(
                prompt=inputs.prompt,
                output_path=path,
                aspect_ratio=inputs.aspect_ratio,
                temperature=inputs.temperature,
                model=resolved_model,
                safety_setting=inputs.safety_setting,
                image_size=resolved_image_size,
                person_generation=inputs.person_generation,
                reference_images=inputs.reference_images,
                output_format=inputs.output_format,
            )

        raw_results = await asyncio.gather(
            *[_generate_one(p) for p in paths],
            return_exceptions=True,
        )

        # Process results
        errors: list[str] = []
        results: list[GenerationResult] = []
        for i, result in enumerate(raw_results):
            if isinstance(result, Exception):
                errors.append(str(result))
                results.append(
                    GenerationResult(success=False, output="", error=str(result))
                )
            else:
                results.append(result)
                if not result.success:
                    errors.append(result.error or "Unknown error")

        # Aggregate: report success based on ALL results
        succeeded = [r for r in results if r.success]
        successful_paths = [p for p, r in zip(paths, results) if r.success]

        if not succeeded:
            return {
                "success": False,
                "message": "All image generations failed",
                "errors": errors,
            }

        response = format_image_result(succeeded[0], successful_paths)
        if errors and n > 1:
            response["partial_failure"] = True
            response["errors"] = errors
            response["message"] = (
                f"{len(succeeded)}/{n} images generated" " successfully"
            )
        return response

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return {"success": False, "message": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"Error generating image: {str(e)}",
        }


@mcp.tool()
async def edit_image(
    prompt: str,
    input_image_path: str,
    output_filename: Optional[str] = None,
    temperature: float = 0.7,
    model: Optional[str] = None,
    output_format: str = "png",
) -> dict:
    """Edit an existing image using a text prompt.

    Args:
        prompt: Description of the desired changes
        input_image_path: Path to the image to edit
        output_filename: Custom filename for edited image (optional)
        temperature: Creativity level 0.0-2.0
        model: Model override. "gemini-3-pro-image-preview" best
            for complex multi-turn edits.
        output_format: File format: "png", "jpeg", or "webp"

    Returns:
        Dictionary with editing result and new image path

    Example:
        edit_image(
            prompt="Add a rainbow in the sky",
            input_image_path="/path/to/image.png",
            model="gemini-3-pro-image-preview"
        )
    """
    logger.info(f"Editing image: {input_image_path}")

    try:
        inputs = EditImageInput(
            prompt=prompt,
            input_image_path=input_image_path,
            output_filename=output_filename,
            temperature=temperature,
            model=model,
            output_format=output_format,
        )

        output_path = generate_output_path(
            filename=inputs.output_filename,
            prefix="edited",
            output_format=inputs.output_format,
        )

        client = get_api_client()
        result = await client.edit(
            prompt=inputs.prompt,
            input_path=Path(inputs.input_image_path),
            output_path=output_path,
            temperature=inputs.temperature,
            model=inputs.model,
            output_format=inputs.output_format,
        )

        response = format_image_result(result, [output_path])
        if result.success:
            logger.info(f"Image edited: {output_path}")
        else:
            logger.error(f"Edit failed: {result.error}")
        return response

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return {"success": False, "message": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"Error editing image: {str(e)}",
        }


@mcp.tool()
async def remove_background(
    image_path: str,
    output_filename: Optional[str] = None,
    output_format: str = "png",
) -> dict:
    """Remove the background from an image.

    Uses AI to isolate the main subject and remove the background.
    Best results with clear subjects against distinct backgrounds.

    Args:
        image_path: Path to the image
        output_filename: Custom filename for result (optional)
        output_format: "png" (default, supports transparency),
            "jpeg", or "webp"

    Returns:
        Dictionary with result and output image path

    Example:
        remove_background(image_path="/path/to/product.jpg")
    """
    logger.info(f"Removing background: {image_path}")

    try:
        inputs = RemoveBackgroundInput(
            image_path=image_path,
            output_filename=output_filename,
            output_format=output_format,
        )

        output_path = generate_output_path(
            filename=inputs.output_filename,
            prefix="nobg",
            output_format=inputs.output_format,
        )

        client = get_api_client()
        result = await client.remove_background(
            image_path=Path(inputs.image_path),
            output_path=output_path,
            output_format=inputs.output_format,
        )

        return format_image_result(result, [output_path])

    except ValidationError as e:
        return {"success": False, "message": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"Error removing background: {str(e)}",
        }


@mcp.tool()
async def transform_image(
    image_path: str,
    operation: str,
    output_filename: Optional[str] = None,
    output_format: str = "png",
    x: Optional[int] = None,
    y: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    maintain_aspect: bool = True,
    degrees: Optional[float] = None,
    direction: Optional[str] = None,
    radius: float = 2.0,
    factor: float = 2.0,
    text: Optional[str] = None,
    position: str = "bottom-right",
    opacity: float = 0.5,
) -> dict:
    """Transform an image using Pillow operations.

    Args:
        image_path: Path to the image to transform
        operation: One of: crop, resize, rotate, flip, blur, sharpen,
            grayscale, watermark
        output_filename: Custom output filename (optional)
        output_format: Output format (png, jpeg, webp)
        x, y: Crop offset coordinates
        width, height: Dimensions for crop/resize
        maintain_aspect: Keep aspect ratio when resizing (default: True)
        degrees: Rotation angle in degrees
        direction: Flip direction ('horizontal' or 'vertical')
        radius: Blur radius (default: 2.0)
        factor: Sharpness factor (default: 2.0, >1 = sharper)
        text: Watermark text
        position: Watermark position (default: 'bottom-right')
        opacity: Watermark opacity 0.0-1.0 (default: 0.5)

    Returns:
        Dictionary with transform result and output path
    """
    logger.info(f"Transforming image: {operation}")

    try:
        inputs = TransformImageInput(
            image_path=image_path,
            operation=operation,
            output_filename=output_filename,
            output_format=output_format,
            x=x,
            y=y,
            width=width,
            height=height,
            maintain_aspect=maintain_aspect,
            degrees=degrees,
            direction=direction,
            radius=radius,
            factor=factor,
            text=text,
            position=position,
            opacity=opacity,
        )

        from PIL import Image as PILImage

        from pixelforge_mcp.utils import transforms

        img = PILImage.open(inputs.image_path)

        # Dispatch to the correct transform function
        op = inputs.operation
        if op == "crop":
            if (
                inputs.x is None
                or inputs.y is None
                or inputs.width is None
                or inputs.height is None
            ):
                return {
                    "success": False,
                    "message": "crop requires x, y, width, height",
                }
            img = transforms.crop(img, inputs.x, inputs.y, inputs.width, inputs.height)
        elif op == "resize":
            if inputs.width is None or inputs.height is None:
                return {
                    "success": False,
                    "message": "resize requires width and height",
                }
            img = transforms.resize(
                img, inputs.width, inputs.height, inputs.maintain_aspect
            )
        elif op == "rotate":
            if inputs.degrees is None:
                return {
                    "success": False,
                    "message": "rotate requires degrees",
                }
            img = transforms.rotate(img, inputs.degrees)
        elif op == "flip":
            if inputs.direction is None:
                return {
                    "success": False,
                    "message": "flip requires direction",
                }
            img = transforms.flip(img, inputs.direction)
        elif op == "blur":
            img = transforms.blur(img, inputs.radius)
        elif op == "sharpen":
            img = transforms.sharpen(img, inputs.factor)
        elif op == "grayscale":
            img = transforms.grayscale(img)
        elif op == "watermark":
            if inputs.text is None:
                return {
                    "success": False,
                    "message": "watermark requires text",
                }
            img = transforms.watermark(
                img, inputs.text, inputs.position, inputs.opacity
            )

        # Save result
        output_path = generate_output_path(
            filename=inputs.output_filename,
            prefix=f"transformed_{op}",
            output_format=inputs.output_format,
        )
        ImagenAPIClient._save_image(img, output_path, inputs.output_format)

        return {
            "success": True,
            "message": f"Image {op} applied successfully",
            "output_path": str(output_path.absolute()),
            "operation": op,
            "size_bytes": output_path.stat().st_size,
        }

    except ValidationError as e:
        return {"success": False, "message": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Transform error: {e}", exc_info=True)
        return {"success": False, "message": f"Transform failed: {e}"}


# ------------------------------------------------------------------
# Analysis tools
# ------------------------------------------------------------------


@mcp.tool()
async def analyze_image(image_path: str, prompt: Optional[str] = None) -> dict:
    """Analyze an image and get a detailed description.

    Args:
        image_path: Path to the image to analyze
        prompt: Custom analysis prompt. Default: general description.

    Returns:
        Dictionary with analysis results

    Example:
        analyze_image(image_path="/path/to/photo.jpg")

        analyze_image(
            image_path="/path/to/photo.jpg",
            prompt="Write an alt-text description for accessibility"
        )
    """
    logger.info(f"Analyzing image: {image_path}")

    try:
        inputs = AnalyzeImageInput(image_path=image_path, prompt=prompt)

        client = get_api_client()
        result = await client.analyze(Path(inputs.image_path), prompt=inputs.prompt)

        if result.success and result.data:
            return {
                "success": True,
                "analysis": result.data.get("analysis", result.output),
                "image_path": inputs.image_path,
            }
        else:
            return {
                "success": False,
                "message": result.error or "Analysis failed",
            }

    except ValidationError as e:
        return {"success": False, "message": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"Error analyzing image: {str(e)}",
        }


@mcp.tool()
async def extract_text(image_path: str) -> dict:
    """Extract text from an image using OCR.

    Uses Gemini's vision capabilities for high-quality text extraction.
    Works with screenshots, documents, signs, handwriting, etc.

    Args:
        image_path: Path to the image to extract text from

    Returns:
        Dictionary with extracted text and text blocks

    Example:
        extract_text(image_path="/path/to/screenshot.png")
    """
    logger.info(f"Extracting text: {image_path}")

    try:
        inputs = ExtractTextInput(image_path=image_path)

        client = get_api_client()
        result = await client.extract_text(Path(inputs.image_path))

        if result.success and result.data:
            return {
                "success": True,
                "text": result.data.get("extracted_text", ""),
                "blocks": result.data.get("blocks", []),
                "image_path": inputs.image_path,
            }
        else:
            return {
                "success": False,
                "message": result.error or "Text extraction failed",
            }

    except ValidationError as e:
        return {"success": False, "message": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"Error extracting text: {str(e)}",
        }


@mcp.tool()
async def detect_objects(image_path: str, objects: Optional[str] = None) -> dict:
    """Detect objects in an image with bounding boxes.

    Uses Gemini's zero-shot object detection. Returns bounding box
    coordinates normalized to 0-1000 scale as [y_min, x_min, y_max,
    x_max].

    Args:
        image_path: Path to the image
        objects: Specific objects to detect (e.g. "cats and dogs").
            Default: detect all visible objects.

    Returns:
        Dictionary with detected objects and bounding boxes

    Example:
        detect_objects(image_path="/path/to/photo.jpg")

        detect_objects(
            image_path="/path/to/photo.jpg",
            objects="people and cars"
        )
    """
    logger.info(f"Detecting objects: {image_path}")

    try:
        inputs = DetectObjectsInput(image_path=image_path, objects=objects)

        client = get_api_client()
        result = await client.detect_objects(
            Path(inputs.image_path), objects=inputs.objects
        )

        if result.success and result.data:
            return {
                "success": True,
                "detections": result.data.get("detections", []),
                "count": result.data.get("count", 0),
                "image_path": inputs.image_path,
            }
        else:
            return {
                "success": False,
                "message": result.error or "Detection failed",
            }

    except ValidationError as e:
        return {"success": False, "message": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"Error detecting objects: {str(e)}",
        }


@mcp.tool()
async def compare_images(
    image_paths: List[str],
    prompt: Optional[str] = None,
) -> dict:
    """Compare two or more images and analyze differences.

    Useful for A/B testing, design review, before/after comparison,
    and change detection.

    Args:
        image_paths: Paths to 2+ images to compare
        prompt: Comparison focus (e.g. "color differences").
            Default: general comparison.

    Returns:
        Dictionary with comparison analysis

    Example:
        compare_images(
            image_paths=["/path/to/v1.png", "/path/to/v2.png"]
        )

        compare_images(
            image_paths=["/path/a.jpg", "/path/b.jpg"],
            prompt="Which image has better composition?"
        )
    """
    logger.info(f"Comparing {len(image_paths)} images")

    try:
        inputs = CompareImagesInput(image_paths=image_paths, prompt=prompt)

        client = get_api_client()
        result = await client.compare_images(
            [Path(p) for p in inputs.image_paths],
            prompt=inputs.prompt,
        )

        if result.success and result.data:
            data = result.data
            return {
                "success": True,
                "comparison": data.get("analysis", result.output),
                "image_count": len(inputs.image_paths),
            }
        elif result.success:
            return {
                "success": True,
                "comparison": result.output,
                "image_count": len(inputs.image_paths),
            }
        else:
            return {
                "success": False,
                "message": result.error or "Comparison failed",
            }

    except ValidationError as e:
        return {"success": False, "message": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"Error comparing images: {str(e)}",
        }


# ------------------------------------------------------------------
# Utility tools
# ------------------------------------------------------------------


@mcp.tool()
async def optimize_prompt(prompt: str, style: Optional[str] = None) -> dict:
    """Enhance a basic prompt for dramatically better image results.

    Uses AI to add details about lighting, composition, colors,
    textures, and artistic techniques. The #1 way to improve
    generation quality.

    Args:
        prompt: Basic prompt to enhance
        style: Target style (optional): photorealistic, illustration,
            3d_render, pixel_art, watercolor, oil_painting, sketch,
            anime, cinematic, product_photo, architecture, food,
            fashion, abstract

    Returns:
        Dictionary with original and enhanced prompts

    Example:
        optimize_prompt(prompt="a cat sitting on a chair")

        optimize_prompt(
            prompt="coffee shop interior",
            style="cinematic"
        )
    """
    logger.info(f"Optimizing prompt: {prompt[:50]}...")

    try:
        inputs = OptimizePromptInput(prompt=prompt, style=style)

        client = get_api_client()
        result = await client.optimize_prompt(prompt=inputs.prompt, style=inputs.style)

        if result.success and result.data:
            return {
                "success": True,
                "original_prompt": result.data.get("original_prompt", prompt),
                "enhanced_prompt": result.data.get("enhanced_prompt", result.output),
                "style": inputs.style,
            }
        else:
            return {
                "success": False,
                "message": (result.error or "Prompt optimization failed"),
            }

    except ValidationError as e:
        return {"success": False, "message": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"Error optimizing prompt: {str(e)}",
        }


@mcp.tool()
async def list_templates(
    category: Optional[str] = None,
) -> dict:
    """List available prompt templates for image generation.

    Args:
        category: Filter by category (product_photography, social_media,
            illustration, portrait, architecture, food, fashion,
            abstract, logo, panoramic). Omit for all.

    Returns:
        Dictionary with template list and categories
    """
    try:
        inputs = ListTemplatesInput(category=category)

        from pixelforge_mcp.utils.templates import get_template_library

        library = get_template_library()

        templates = library.list_templates(category=inputs.category)
        categories = library.list_categories()

        return {
            "success": True,
            "templates": templates,
            "count": len(templates),
            "categories": categories,
            "filter": inputs.category,
        }
    except ValidationError as e:
        return {"success": False, "message": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Template listing failed: {e}", exc_info=True)
        return {"success": False, "message": str(e)}


@mcp.tool()
async def apply_template(
    template_name: str,
    subject: str,
) -> dict:
    """Apply a prompt template with a subject to generate a ready-to-use prompt.

    Args:
        template_name: Name of the template (e.g., 'product_hero',
            'instagram_post'). Use list_templates() to see all.
        subject: The subject to fill into the template
            (e.g., 'wireless headphones', 'coffee shop')

    Returns:
        Dictionary with rendered prompt and recommendations

    Example:
        apply_template(
            template_name="product_hero",
            subject="wireless headphones"
        )
    """
    try:
        inputs = ApplyTemplateInput(
            template_name=template_name,
            subject=subject,
        )

        from pixelforge_mcp.utils.templates import get_template_library

        library = get_template_library()

        result = library.apply_template(
            inputs.template_name,
            {"subject": inputs.subject},
        )

        if result:
            return {
                "success": True,
                **result,
            }
        else:
            available = [t["name"] for t in library.list_templates()]
            return {
                "success": False,
                "message": f"Template '{inputs.template_name}' not found",
                "available_templates": available,
            }
    except ValidationError as e:
        return {"success": False, "message": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Template application failed: {e}", exc_info=True)
        return {"success": False, "message": str(e)}


@mcp.tool()
async def estimate_cost(
    operation: str,
    model: Optional[str] = None,
    number_of_images: int = 1,
) -> dict:
    """Estimate the cost of an image operation.

    Shows pricing per model per operation to help with budgeting.

    Args:
        operation: "generate", "edit", or "analyze"
        model: Model to estimate for (default: all models)
        number_of_images: Number of images (default: 1)

    Returns:
        Dictionary with cost estimates

    Example:
        estimate_cost(operation="generate")
        estimate_cost(
            operation="generate",
            model="gemini-3-pro-image-preview",
            number_of_images=10
        )
    """
    try:
        inputs = EstimateCostInput(
            operation=operation,
            model=model,
            number_of_images=number_of_images,
        )

        if inputs.model:
            # Single model estimate
            costs = COST_TABLE.get(inputs.model)
            if not costs:
                return {
                    "success": False,
                    "message": f"Unknown model: {inputs.model}",
                }
            unit_cost = costs.get(inputs.operation, 0)
            total = unit_cost * inputs.number_of_images
            return {
                "success": True,
                "model": inputs.model,
                "operation": inputs.operation,
                "unit_cost_usd": unit_cost,
                "number_of_images": inputs.number_of_images,
                "total_cost_usd": round(total, 4),
            }
        else:
            # All models comparison
            estimates = []
            for model_name, costs in COST_TABLE.items():
                unit_cost = costs.get(inputs.operation, 0)
                total = unit_cost * inputs.number_of_images
                estimates.append(
                    {
                        "model": model_name,
                        "unit_cost_usd": unit_cost,
                        "total_cost_usd": round(total, 4),
                    }
                )
            return {
                "success": True,
                "operation": inputs.operation,
                "number_of_images": inputs.number_of_images,
                "estimates": estimates,
                "note": "Prices are approximate and may change.",
            }

    except ValidationError as e:
        return {"success": False, "message": f"Invalid input: {e}"}
    except Exception as e:
        return {
            "success": False,
            "message": f"Error estimating cost: {str(e)}",
        }


# ------------------------------------------------------------------
# Info tools
# ------------------------------------------------------------------


@mcp.tool()
async def list_available_models() -> dict:
    """List all available image generation models with capabilities.

    Returns model details including speed, quality, resolution
    support, and best use cases. Use this to pick the right model
    for your task.

    Returns:
        Dictionary with model details and selection guidance

    Example:
        list_available_models()
    """
    logger.info("Listing available models")

    try:
        client = get_api_client()
        result = await client.list_models()

        if result.success and result.data:
            return {
                "success": True,
                "models": result.data.get("models", []),
                "recommendation": result.data.get("recommendation", ""),
                "note": (
                    "Each model can be selected per-request "
                    "via the 'model' parameter in "
                    "generate_image() and edit_image()"
                ),
            }
        else:
            return {
                "success": True,
                "models": [
                    {
                        "name": "gemini-2.5-flash-image",
                        "description": "Fast model for iterations",
                        "default": True,
                    },
                    {
                        "name": "gemini-3.1-flash-image-preview",
                        "description": "Panoramic, grounded, 4K",
                        "default": False,
                    },
                    {
                        "name": "gemini-3-pro-image-preview",
                        "description": "Max text fidelity",
                        "default": False,
                    },
                ],
                "note": "Model switching available on every request",
            }

    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"Error listing models: {str(e)}",
        }


@mcp.tool()
def get_server_info() -> dict:
    """Get MCP server configuration, capabilities, and available tools.

    Returns:
        Dictionary with server info and feature overview
    """
    config = get_config()

    return {
        "server": {
            "name": config.server.name,
            "version": config.server.version,
            "log_level": config.server.log_level,
        },
        "storage": {
            "output_directory": str(config.storage.output_dir),
        },
        "imagen": {
            "default_model": config.imagen.default_model,
            "default_aspect_ratio": (config.imagen.default_aspect_ratio),
            "default_temperature": config.imagen.default_temperature,
        },
        "tools": {
            "generation": [
                "generate_image",
                "edit_image",
                "remove_background",
                "transform_image",
            ],
            "analysis": [
                "analyze_image",
                "extract_text",
                "detect_objects",
                "compare_images",
            ],
            "utility": [
                "optimize_prompt",
                "list_templates",
                "apply_template",
                "estimate_cost",
                "list_available_models",
                "get_server_info",
            ],
        },
        "features": {
            "resolution_control": "1K/2K/4K via image_size",
            "multi_image": "1-4 variations via number_of_images",
            "output_formats": "png, jpeg, webp",
            "person_generation": "allow, adults_only, block",
            "reference_images": "Up to 14 style/character refs",
            "prompt_optimization": "AI-enhanced prompts",
            "ocr": "Text extraction from images",
            "object_detection": "Bounding box detection",
            "background_removal": "Subject isolation",
            "image_transforms": "Crop, resize, rotate, flip, "
            "blur, sharpen, grayscale, watermark",
            "image_comparison": "Multi-image analysis",
            "cost_estimation": "Per-operation pricing",
            "prompt_templates": "Curated templates with " "model/ratio recommendations",
        },
    }


def main():
    """Main entry point for the MCP server."""
    try:
        config = get_config()
        logging.getLogger().setLevel(config.server.log_level)
        logger.info(f"Starting {config.server.name} v{config.server.version}")
        logger.info(f"Output directory: {config.storage.output_dir}")
        mcp.run()
    except Exception as e:
        logger.error(f"Server startup failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
