"""Main MCP server implementation for Gemini Imagen."""

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
    AnalyzeImageInput,
    CompareImagesInput,
    DetectObjectsInput,
    EditImageInput,
    EstimateCostInput,
    ExtractTextInput,
    GenerateImageInput,
    OptimizePromptInput,
    RemoveBackgroundInput,
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
        _client = ImagenAPIClient(
            model_name=config.imagen.default_model,
            api_key=config.imagen.api_key,
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


def format_api_result(
    result: GenerationResult,
    image_path: Optional[Path] = None,
) -> dict:
    """Format API result for MCP response."""
    response = {
        "success": result.success,
        "message": result.output if result.success else result.error,
    }

    if result.success and image_path and image_path.exists():
        response["image_path"] = str(image_path.absolute())
        response["image_size_bytes"] = image_path.stat().st_size

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
            safety_setting=safety_setting,
            image_size=image_size,
            number_of_images=number_of_images,
            output_format=output_format,
            person_generation=person_generation,
            reference_images=reference_images,
        )

        client = get_api_client()

        # Single image (most common path)
        if inputs.number_of_images == 1:
            output_path = generate_output_path(
                filename=inputs.output_filename,
                prefix="generated",
                output_format=inputs.output_format,
            )

            result = await client.generate(
                prompt=inputs.prompt,
                output_path=output_path,
                aspect_ratio=inputs.aspect_ratio,
                temperature=inputs.temperature,
                model=inputs.model,
                safety_setting=inputs.safety_setting,
                image_size=inputs.image_size,
                person_generation=inputs.person_generation,
                reference_images=inputs.reference_images,
                output_format=inputs.output_format,
            )

            response = format_api_result(
                result, output_path if result.success else None
            )
            if result.success:
                logger.info(f"Image generated: {output_path}")
            else:
                logger.error(f"Generation failed: {result.error}")
            return response

        # Multiple images
        generated = []
        errors = []
        for i in range(inputs.number_of_images):
            suffix = f"_{i + 1}"
            if inputs.output_filename:
                base, ext = inputs.output_filename.rsplit(".", 1)
                numbered_name = f"{base}{suffix}.{ext}"
            else:
                numbered_name = None
            output_path = generate_output_path(
                filename=numbered_name,
                prefix=f"generated{suffix}",
                output_format=inputs.output_format,
            )

            result = await client.generate(
                prompt=inputs.prompt,
                output_path=output_path,
                aspect_ratio=inputs.aspect_ratio,
                temperature=inputs.temperature,
                model=inputs.model,
                safety_setting=inputs.safety_setting,
                image_size=inputs.image_size,
                person_generation=inputs.person_generation,
                reference_images=inputs.reference_images,
                output_format=inputs.output_format,
            )

            if result.success and output_path.exists():
                generated.append(
                    {
                        "image_path": str(output_path.absolute()),
                        "image_size_bytes": output_path.stat().st_size,
                    }
                )
            else:
                errors.append(result.error or "Unknown error")

        if generated:
            response = {
                "success": True,
                "message": (
                    f"Generated {len(generated)} of "
                    f"{inputs.number_of_images} images"
                ),
                "image_path": generated[0]["image_path"],
                "image_size_bytes": generated[0]["image_size_bytes"],
                "images": generated,
                "details": {
                    "model": inputs.model,
                    "aspect_ratio": inputs.aspect_ratio,
                    "temperature": inputs.temperature,
                    "image_size": inputs.image_size,
                    "count": len(generated),
                },
            }
            if errors:
                response["errors"] = errors
            logger.info(f"Generated {len(generated)} images")
            return response
        else:
            return {
                "success": False,
                "message": (f"All {inputs.number_of_images} " f"generations failed"),
                "errors": errors,
            }

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

        response = format_api_result(result, output_path if result.success else None)
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

        return format_api_result(result, output_path if result.success else None)

    except ValidationError as e:
        return {"success": False, "message": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"Error removing background: {str(e)}",
        }


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

        if result.success:
            return {
                "success": True,
                "comparison": result.data.get("analysis", result.output),
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
            "use_s3": config.storage.use_s3,
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
            ],
            "analysis": [
                "analyze_image",
                "extract_text",
                "detect_objects",
                "compare_images",
            ],
            "utility": [
                "optimize_prompt",
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
            "image_comparison": "Multi-image analysis",
            "cost_estimation": "Per-operation pricing",
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
