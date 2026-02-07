"""Main MCP server implementation for Gemini Imagen."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP
from pydantic import ValidationError

from .config import get_config
from .utils.api_client import ImagenAPIClient, GenerationResult
from .utils.validation import (
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
    filename: Optional[str] = None, prefix: str = "generated"
) -> Path:
    """Generate a unique output path for an image.

    Args:
        filename: Custom filename (optional)
        prefix: Prefix for auto-generated filenames

    Returns:
        Path object for the output file
    """
    config = get_config()
    output_dir = config.storage.output_dir

    if filename:
        return output_dir / filename

    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return output_dir / f"{prefix}_{timestamp}.png"


def format_api_result(result: GenerationResult, image_path: Optional[Path] = None) -> dict:
    """Format API result for MCP response.

    Args:
        result: API execution result
        image_path: Path to generated/edited image

    Returns:
        Formatted response dictionary
    """
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
) -> dict:
    """Generate an image from a text prompt using Google Gemini.

    Args:
        prompt: Text description of the image to generate
        output_filename: Custom filename (optional, will auto-generate if not provided)
        aspect_ratio: Image dimensions (1:1, 16:9, 9:16, etc.)
        temperature: Creativity level 0.0-1.0 (higher = more creative)
        model: Specific model to use (optional). Available models:
            - "gemini-2.5-flash-image" (default): Fast generation, good for iterations
            - "gemini-3-pro-image-preview": Highest quality, best for final outputs
            Call list_available_models() for detailed model capabilities.
        safety_setting: Content safety filter (preset:strict, preset:relaxed)

    Returns:
        Dictionary with generation result and image path

    MODEL SWITCHING GUIDANCE:
        Choose model based on your needs:
        - Speed/iteration: Use gemini-2.5-flash-image (default)
        - Quality/complexity: Use gemini-3-pro-image-preview
        - Text in images: Use gemini-3-pro-image-preview
        - High resolution (2K/4K): Use gemini-3-pro-image-preview

        You can switch models on every request - no setup needed!

    Example:
        # Fast iteration
        generate_image(
            prompt="Quick concept sketch",
            model="gemini-2.5-flash-image"
        )

        # High quality final
        generate_image(
            prompt="Photorealistic portrait with intricate details",
            model="gemini-3-pro-image-preview",
            aspect_ratio="16:9",
            temperature=0.8
        )
    """
    logger.info(f"Generating image: {prompt[:50]}...")

    try:
        # Validate inputs
        inputs = GenerateImageInput(
            prompt=prompt,
            output_filename=output_filename,
            aspect_ratio=aspect_ratio,
            temperature=temperature,
            model=model,
            safety_setting=safety_setting,
        )

        # Generate output path
        output_path = generate_output_path(
            filename=inputs.output_filename, prefix="generated"
        )

        # Execute generation
        client = get_api_client()
        result = await client.generate(
            prompt=inputs.prompt,
            output_path=output_path,
            aspect_ratio=inputs.aspect_ratio,
            temperature=inputs.temperature,
            model=inputs.model,
            safety_setting=inputs.safety_setting,
        )

        # Format and return result
        response = format_api_result(result, output_path if result.success else None)

        if result.success:
            logger.info(f"Image generated successfully: {output_path}")
        else:
            logger.error(f"Image generation failed: {result.error}")

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
            "message": f"Error generating image: {str(e)}",
        }


@mcp.tool()
async def edit_image(
    prompt: str,
    input_image_path: str,
    output_filename: Optional[str] = None,
    temperature: float = 0.7,
) -> dict:
    """Edit an existing image using a text prompt.

    Args:
        prompt: Description of the desired changes
        input_image_path: Path to the image to edit
        output_filename: Custom filename for edited image (optional)
        temperature: Creativity level 0.0-1.0

    Returns:
        Dictionary with editing result and new image path

    Example:
        edit_image(
            prompt="Add a rainbow in the sky",
            input_image_path="/path/to/image.png",
            temperature=0.8
        )
    """
    logger.info(f"Editing image: {input_image_path}")

    try:
        # Validate inputs
        inputs = EditImageInput(
            prompt=prompt,
            input_image_path=input_image_path,
            output_filename=output_filename,
            temperature=temperature,
        )

        # Generate output path
        output_path = generate_output_path(
            filename=inputs.output_filename, prefix="edited"
        )

        # Execute editing
        client = get_api_client()
        result = await client.edit(
            prompt=inputs.prompt,
            input_path=Path(inputs.input_image_path),
            output_path=output_path,
            temperature=inputs.temperature,
        )

        # Format and return result
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
async def analyze_image(image_path: str) -> dict:
    """Analyze an image and get a detailed description.

    Args:
        image_path: Path to the image to analyze

    Returns:
        Dictionary with analysis results

    Example:
        analyze_image(image_path="/path/to/photo.jpg")
    """
    logger.info(f"Analyzing image: {image_path}")

    try:
        # Validate inputs
        inputs = AnalyzeImageInput(image_path=image_path)

        # Execute analysis
        client = get_api_client()
        result = await client.analyze(Path(inputs.image_path))

        # Format response
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
    - Specific capabilities (text rendering, resolution, etc.)

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
                "note": "Each model can be selected per-request using the 'model' parameter in generate_image()"
            }
        else:
            # Fallback - basic model list
            return {
                "success": True,
                "models": [
                    {
                        "name": "gemini-2.5-flash-image",
                        "description": "Fast model for iterations",
                        "default": True
                    },
                    {
                        "name": "gemini-3-pro-image-preview",
                        "description": "High quality model for final outputs",
                        "default": False
                    }
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
            "available_models": 2,
            "guidance": "Models can be switched on every generate_image() call. Use 'model' parameter to override default. Call list_available_models() for detailed capabilities.",
            "quick_tips": {
                "fast": "gemini-2.5-flash-image",
                "quality": "gemini-3-pro-image-preview",
                "text_in_images": "gemini-3-pro-image-preview",
                "high_res": "gemini-3-pro-image-preview"
            }
        },
    }


def main():
    """Main entry point for the MCP server."""
    try:
        # Load configuration
        config = get_config()

        # Set logging level
        logging.getLogger().setLevel(config.server.log_level)

        logger.info(f"Starting {config.server.name} v{config.server.version}")
        logger.info(f"Output directory: {config.storage.output_dir}")

        # Run the server
        mcp.run()

    except Exception as e:
        logger.error(f"Server startup failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
