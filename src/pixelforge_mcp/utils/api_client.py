"""API client for gemini-imagen library."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from gemini_imagen import GeminiImageGenerator
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_ANALYSIS_PROMPT = (
    "Describe this image in detail, including objects, colors, composition, and mood."
)


@dataclass
class GenerationResult:
    """Result from image generation/editing/analysis."""

    success: bool
    output: str
    error: Optional[str] = None
    images: Optional[List[Image.Image]] = None
    image_paths: Optional[List[str]] = None
    data: Optional[Dict[str, Any]] = None


class ImagenAPIClient:
    """Client for gemini-imagen Python API."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-image",
        api_key: Optional[str] = None,
        log_images: bool = False,
    ):
        """Initialize API client.

        Args:
            model_name: Gemini model to use for generation
            api_key: Google API key (loads from env if not provided)
            log_images: Enable LangSmith tracing
        """
        self.model_name = model_name
        self.api_key = api_key
        self.log_images = log_images
        self._generator: Optional[GeminiImageGenerator] = None

    def _get_generator(self) -> GeminiImageGenerator:
        """Get or create generator instance."""
        if self._generator is None:
            self._generator = GeminiImageGenerator(
                model_name=self.model_name,
                api_key=self.api_key,
                log_images=self.log_images,
            )
        return self._generator

    async def generate(
        self,
        prompt: str,
        output_path: Path,
        aspect_ratio: Optional[str] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        safety_setting: Optional[str] = None,
    ) -> GenerationResult:
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the image
            output_path: Where to save the generated image
            aspect_ratio: Image aspect ratio (e.g., "16:9")
            temperature: Generation temperature (0-1)
            model: Model to use for generation
            safety_setting: Safety filter setting

        Returns:
            GenerationResult with generation details
        """
        try:
            # Use specified model or default
            generator = self._get_generator()
            if model:
                generator.model_name = model

            # Call the API
            result = await generator.generate(
                prompt=prompt,
                output_images=[str(output_path)],
                aspect_ratio=aspect_ratio,
                temperature=temperature,
            )

            # Format response
            if result.images:
                return GenerationResult(
                    success=True,
                    output=f"Image generated successfully at {output_path}",
                    images=result.images,
                    image_paths=[str(output_path)],
                    data={
                        "model": model or self.model_name,
                        "aspect_ratio": aspect_ratio,
                        "temperature": temperature,
                    },
                )
            else:
                return GenerationResult(
                    success=False,
                    output="",
                    error="No images generated",
                )

        except Exception as e:
            logger.error(f"Image generation failed: {e}", exc_info=True)
            return GenerationResult(
                success=False,
                output="",
                error=str(e),
            )

    async def edit(
        self,
        prompt: str,
        input_path: Path,
        output_path: Path,
        temperature: Optional[float] = None,
    ) -> GenerationResult:
        """Edit an existing image with a text prompt.

        Args:
            prompt: Text description of desired changes
            input_path: Path to the input image
            output_path: Where to save the edited image
            temperature: Generation temperature (0-1)

        Returns:
            GenerationResult with editing details
        """
        try:
            generator = self._get_generator()

            result = await generator.generate(
                prompt=prompt,
                input_images=[str(input_path)],
                output_images=[str(output_path)],
                temperature=temperature,
            )

            if result.images:
                return GenerationResult(
                    success=True,
                    output=f"Image edited successfully at {output_path}",
                    images=result.images,
                    image_paths=[str(output_path)],
                    data={
                        "model": self.model_name,
                        "temperature": temperature,
                    },
                )
            else:
                return GenerationResult(
                    success=False,
                    output="",
                    error="No images generated",
                )

        except Exception as e:
            logger.error(f"Image editing failed: {e}", exc_info=True)
            return GenerationResult(
                success=False,
                output="",
                error=str(e),
            )

    async def analyze(
        self, image_path: Path, prompt: Optional[str] = None
    ) -> GenerationResult:
        """Analyze an image and get a description.

        Args:
            image_path: Path to the image to analyze
            prompt: Custom analysis prompt (uses DEFAULT_ANALYSIS_PROMPT if None)

        Returns:
            GenerationResult with analysis data
        """
        try:
            generator = self._get_generator()

            result = await generator.generate(
                prompt=prompt or DEFAULT_ANALYSIS_PROMPT,
                input_images=[str(image_path)],
                output_text=True,
            )

            if result.text:
                return GenerationResult(
                    success=True,
                    output=result.text,
                    data={
                        "analysis": result.text,
                        "image_path": str(image_path),
                    },
                )
            else:
                return GenerationResult(
                    success=False,
                    output="",
                    error="No analysis generated",
                )

        except Exception as e:
            logger.error(f"Image analysis failed: {e}", exc_info=True)
            return GenerationResult(
                success=False,
                output="",
                error=str(e),
            )

    async def list_models(self) -> GenerationResult:
        """List available generation models with detailed metadata.

        Returns:
            GenerationResult with model list and capabilities
        """
        # Model metadata with capabilities and use cases
        models = [
            {
                "name": "gemini-2.5-flash-image",
                "nickname": "Nano Banana",
                "speed": "fast",
                "quality": "good",
                "default": True,
                "description": "Optimized for speed and efficiency. Best for high-volume, low-latency tasks.",
                "best_for": [
                    "Quick iterations and concept exploration",
                    "High-volume batch generation",
                    "Simple compositions and designs",
                    "When speed matters more than perfection",
                ],
                "capabilities": {
                    "text_rendering": "basic",
                    "complex_scenes": "moderate",
                    "editing": "basic",
                    "resolution": "1K",
                },
                "temperature": {
                    "min": 0.0,
                    "max": 2.0,
                    "default": 1.0,
                },
            },
            {
                "name": "gemini-3-pro-image-preview",
                "nickname": "Nano Banana Pro",
                "speed": "moderate",
                "quality": "excellent",
                "default": False,
                "description": "Max text fidelity (~94%), "
                "deep reasoning for complex prompts.",
                "best_for": [
                    "Highest text rendering accuracy (~94%)",
                    "Complex multi-turn editing workflows",
                    "Deep reasoning on intricate prompts",
                    "Character consistency (up to 5 chars)",
                ],
                "capabilities": {
                    "text_rendering": "excellent (~94%)",
                    "complex_scenes": "excellent",
                    "editing": "advanced multi-turn",
                    "resolution": "1K/2K/4K",
                    "reference_inputs": "6 objects + 5 characters",
                    "thinking_process": True,
                    "grounding": "web search",
                },
                "temperature": {
                    "min": 0.0,
                    "max": 2.0,
                    "default": 1.0,
                },
            },
            {
                "name": "gemini-3.1-flash-image-preview",
                "nickname": "Nano Banana 2",
                "speed": "fast",
                "quality": "excellent",
                "default": False,
                "description": "Excellent quality at Flash speed. "
                "Panoramic ratios, web+image grounding, 512px tier.",
                "best_for": [
                    "Panoramic aspect ratios (1:4, 4:1, 1:8, 8:1)",
                    "Grounded generation (web + Google Image Search)",
                    "Fast 4K output (4-6s vs Pro's 8-12s)",
                    "512px rapid iteration tier",
                    "Cost-sensitive high-volume generation",
                ],
                "capabilities": {
                    "text_rendering": "excellent",
                    "resolution": "512px/1K/2K/4K",
                    "panoramic_ratios": True,
                    "reference_inputs": "up to 14 objects + 5 characters",
                    "thinking_process": True,
                    "grounding": "web search + Google Image Search",
                },
                "temperature": {
                    "min": 0.0,
                    "max": 2.0,
                    "default": 1.0,
                },
            },
        ]

        return GenerationResult(
            success=True,
            output=f"Found {len(models)} available models",
            data={
                "models": models,
                "recommendation": (
                    "Use gemini-2.5-flash-image for speed, "
                    "gemini-3-pro-image-preview for quality, "
                    "gemini-3.1-flash-image-preview for "
                    "quality+speed and panoramic ratios"
                ),
            },
        )
