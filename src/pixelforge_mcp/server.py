"""Main MCP server implementation for Gemini Imagen."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP
from pydantic import ValidationError

from .config import get_config
from .utils.api_client import GenerationResult, ImagenAPIClient
from .utils.validation import (
    FORMAT_EXTENSIONS,
    AnalyzeImageInput,
    EditImageInput,
    GenerateImageInput,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
    """Generate a unique output path for an image.

    Args:
        filename: Custom filename (optional)
        prefix: Prefix for auto-generated filenames
        output_format: Image format for extension (png, jpeg, webp)

    Returns:
        Path object for the output file
    """
    config = get_config()
    output_dir = config.storage.output_dir

    if filename:
        return output_dir / filename

    ext = FORMAT_EXTENSIONS.get(output_format, ".png")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return output_dir / f"{prefix}_{timestamp}{ext}"


def format_api_result(
    result: GenerationResult, image_path: Optional[Path] = None
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
) -> dict:
    """Generate an image from a text prompt using Google Gemini.

    Args:
        prompt: Text description of the image to generate
        output_filename: Custom filename (optional, auto-generated if not provided)
        aspect_ratio: Image dimensions (1:1, 16:9, 9:16, 4:3, 3:4, etc.)
        temperature: Creativity level 0.0-2.0 (higher = more creative)
        model: Model override. Options:
            - "gemini-2.5-flash-image" (default): Fast, cheap
            - "gemini-3-pro-image-preview": Best text, complex edits
            - "gemini-3.1-flash-image-preview": Panoramic, fast 4K
        safety_setting: Safety filter (preset:strict, preset:relaxed)
        image_size: Output resolution: "1K" (default), "2K", or "4K".
            4K requires gemini-3.1-flash or gemini-3-pro models.
        number_of_images: Generate 1-4 variations (default: 1)
        output_format: File format: "png" (default), "jpeg", or "webp"
        person_generation: Person generation control:
            "allow_all", "allow_adult", or "dont_allow"

    Returns:
        Dictionary with generation result, image path(s), and details

    Example:
        # Simple generation
        generate_image(prompt="A sunset over mountains")

        # High-res with specific format
        generate_image(
            prompt="Product photo of a watch",
            model="gemini-3.1-flash-image-preview",
            image_size="4K",
            output_format="webp"
        )

        # Multiple variations
        generate_image(
            prompt="Logo design for a coffee shop",
            number_of_images=4,
            aspect_ratio="1:1"
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
                output_format=inputs.output_format,
            )

            response = format_api_result(
                result, output_path if result.success else None
            )
            if result.success:
                logger.info(f"Image generated successfully: {output_path}")
            else:
                logger.error(f"Image generation failed: {result.error}")
            return response

        # Multiple images
        generated = []
        errors = []
        for i in range(inputs.number_of_images):
            suffix = f"_{i + 1}"
            output_path = generate_output_path(
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
                "message": f"Generated {len(generated)} of "
                f"{inputs.number_of_images} images",
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
                "message": f"All {inputs.number_of_images} generations failed",
                "errors": errors,
            }

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return {
            "success": False,
            "message": f"Invalid input: {e}",
        }
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
        model: Model override. "gemini-3-pro-image-preview" recommended
            for complex multi-turn edits.
        output_format: File format: "png" (default), "jpeg", or "webp"

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
            logger.info(f"Image edited successfully: {output_path}")
        else:
            logger.error(f"Image editing failed: {result.error}")

        return response

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return {
            "success": False,
            "message": f"Invalid input: {e}",
        }
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"Error editing image: {str(e)}",
        }


@mcp.tool()
async def analyze_image(image_path: str, prompt: Optional[str] = None) -> dict:
    """Analyze an image and get a detailed description.

    Args:
        image_path: Path to the image to analyze
        prompt: Custom analysis prompt. If not provided, returns a general
            description of the image. Use this to focus the analysis on
            specific aspects.

    Returns:
        Dictionary with analysis results

    Example:
        # General description (default)
        analyze_image(image_path="/path/to/photo.jpg")

        # OCR / text extraction
        analyze_image(
            image_path="/path/to/screenshot.png",
            prompt="Extract all visible text from this image"
        )

        # Accessibility description
        analyze_image(
            image_path="/path/to/photo.jpg",
            prompt="Write an alt-text description for accessibility"
        )

        # Color palette extraction
        analyze_image(
            image_path="/path/to/design.png",
            prompt="List the dominant colors and their approximate hex values"
        )
    """
    logger.info(f"Analyzing image: {image_path}")

    try:
        inputs = AnalyzeImageInput(image_path=image_path, prompt=prompt)

        client = get_api_client()
        result = await client.analyze(Path(inputs.image_path), prompt=inputs.prompt)

        if result.success and result.data:
            response = {
                "success": True,
                "analysis": result.data.get("analysis", result.output),
                "image_path": inputs.image_path,
            }
        else:
            response = {
                "success": False,
                "message": result.error or "Analysis failed",
            }

        return response

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return {
            "success": False,
            "message": f"Invalid input: {e}",
        }
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"Error analyzing image: {str(e)}",
        }


@mcp.tool()
async def list_available_models() -> dict:
    """List all available image generation models with detailed capabilities.

    Returns detailed information about each model including:
    - Model name and nickname
    - Speed and quality ratings
    - Best use cases and when to choose each model
    - Specific capabilities (text rendering, resolution, person generation, etc.)

    Use this tool to help decide which model to use for different tasks.
    Models can be switched on every generate_image() call via the 'model' parameter.

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
                    "Each model can be selected per-request using "
                    "the 'model' parameter in generate_image() "
                    "and edit_image()"
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
                        "description": "Panoramic, grounded, fast 4K",
                        "default": False,
                    },
                    {
                        "name": "gemini-3-pro-image-preview",
                        "description": "Max text fidelity, complex edits",
                        "default": False,
                    },
                ],
                "note": "Default models - model switching available on every request",
            }

    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"Error listing models: {str(e)}",
        }


@mcp.tool()
def get_server_info() -> dict:
    """Get information about the MCP server configuration and capabilities.

    Returns:
        Dictionary with server configuration details and model switching guidance
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
            "default_aspect_ratio": config.imagen.default_aspect_ratio,
            "default_temperature": config.imagen.default_temperature,
        },
        "model_switching": {
            "enabled": True,
            "method": "per_request_parameter",
            "available_models": 3,
            "guidance": (
                "Models can be switched on every "
                "generate_image() and edit_image() call. "
                "Use 'model' parameter to override default. "
                "Call list_available_models() for details."
            ),
            "quick_tips": {
                "fast_iterations": "gemini-2.5-flash-image",
                "panoramic_grounded": "gemini-3.1-flash-image-preview",
                "max_text_fidelity": "gemini-3-pro-image-preview",
                "complex_editing": "gemini-3-pro-image-preview",
            },
        },
        "features": {
            "resolution_control": "1K/2K/4K via image_size parameter",
            "multi_image": "1-4 variations via number_of_images parameter",
            "output_formats": "png, jpeg, webp via output_format parameter",
            "person_generation": "allow_all, allow_adult, dont_allow",
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
