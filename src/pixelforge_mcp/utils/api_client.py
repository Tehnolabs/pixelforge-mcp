"""API client for gemini-imagen library and direct google-genai SDK."""

import logging
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from gemini_imagen import GeminiImageGenerator
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_ANALYSIS_PROMPT = (
    "Describe this image in detail, including objects, colors, composition, and mood."
)

# Safety presets — maps user-friendly names to SDK threshold names
SAFETY_PRESETS = {
    "strict": "BLOCK_LOW_AND_ABOVE",
    "relaxed": "BLOCK_ONLY_HIGH",
    "none": "BLOCK_NONE",
}

# Pillow format names for save()
PILLOW_FORMAT_MAP = {
    "png": "PNG",
    "jpeg": "JPEG",
    "webp": "WEBP",
}


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
    """Client for image generation via gemini-imagen and direct google-genai SDK.

    Uses gemini-imagen for basic calls (proven path, LangSmith tracing).
    Uses google-genai SDK directly when extended ImageConfig params are needed
    (image_size, person_generation) — bypasses gemini-imagen limitations.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-image",
        api_key: Optional[str] = None,
        log_images: bool = False,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.log_images = log_images
        self._generator: Optional[GeminiImageGenerator] = None
        self._genai_client: Optional[Any] = None

    def _get_generator(self) -> GeminiImageGenerator:
        """Get or create gemini-imagen generator instance."""
        if self._generator is None:
            self._generator = GeminiImageGenerator(
                model_name=self.model_name,
                api_key=self.api_key,
                log_images=self.log_images,
            )
        return self._generator

    def _get_genai_client(self) -> Any:
        """Get or create direct google-genai client for extended features."""
        if self._genai_client is None:
            from google import genai

            api_key = (
                self.api_key
                or os.getenv("GOOGLE_API_KEY")
                or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
                or os.getenv("GEMINI_API_KEY")
            )
            if not api_key:
                raise ValueError(
                    "No API key found. Set GOOGLE_API_KEY environment variable."
                )
            self._genai_client = genai.Client(api_key=api_key)
        return self._genai_client

    @staticmethod
    def _needs_direct_sdk(
        image_size: Optional[str] = None,
        person_generation: Optional[str] = None,
    ) -> bool:
        """Check if the call requires direct SDK (extended ImageConfig params)."""
        return image_size is not None or person_generation is not None

    @staticmethod
    def _build_safety_settings(
        safety_setting: Optional[str],
    ) -> Optional[list]:
        """Convert preset string to SDK SafetySetting objects."""
        if not safety_setting or not safety_setting.startswith("preset:"):
            return None

        from google.genai import types

        preset_name = safety_setting[len("preset:") :]
        threshold_name = SAFETY_PRESETS.get(preset_name)
        if not threshold_name:
            return None

        threshold = getattr(types.HarmBlockThreshold, threshold_name, None)
        if not threshold:
            return None

        categories = [
            types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        ]
        return [
            types.SafetySetting(category=cat, threshold=threshold) for cat in categories
        ]

    @staticmethod
    def _save_image(
        image: Image.Image,
        output_path: Path,
        output_format: str = "png",
    ) -> None:
        """Save a PIL Image in the specified format."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pillow_format = PILLOW_FORMAT_MAP.get(output_format, "PNG")

        save_kwargs: Dict[str, Any] = {"format": pillow_format}
        if pillow_format == "JPEG":
            if image.mode in ("RGBA", "LA", "PA"):
                image = image.convert("RGB")
            save_kwargs["quality"] = 95

        image.save(str(output_path), **save_kwargs)

    async def _generate_via_sdk(
        self,
        prompt: str,
        output_path: Path,
        model: str,
        aspect_ratio: Optional[str] = None,
        temperature: Optional[float] = None,
        safety_setting: Optional[str] = None,
        image_size: Optional[str] = None,
        person_generation: Optional[str] = None,
        output_format: str = "png",
    ) -> GenerationResult:
        """Generate using google-genai SDK directly for extended ImageConfig."""
        try:
            from google.genai import types

            from .validation import PERSON_GENERATION_SDK_MAP

            client = self._get_genai_client()

            # Build ImageConfig with all supported params
            image_config_params: Dict[str, Any] = {}
            if aspect_ratio:
                image_config_params["aspect_ratio"] = aspect_ratio
            if image_size:
                image_config_params["image_size"] = image_size
            if person_generation:
                sdk_value = PERSON_GENERATION_SDK_MAP.get(person_generation)
                if sdk_value:
                    image_config_params["person_generation"] = sdk_value

            # Build GenerateContentConfig
            config_params: Dict[str, Any] = {
                "response_modalities": ["IMAGE"],
            }
            if temperature is not None:
                config_params["temperature"] = temperature
            if image_config_params:
                config_params["image_config"] = types.ImageConfig(**image_config_params)

            safety_settings = self._build_safety_settings(safety_setting)
            if safety_settings:
                config_params["safety_settings"] = safety_settings

            config = types.GenerateContentConfig(**config_params)

            # Call Gemini API
            response = await client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )

            # Extract image from response
            images: List[Image.Image] = []
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "inline_data") and part.inline_data:
                        img = Image.open(BytesIO(part.inline_data.data))
                        images.append(img)

            if images:
                self._save_image(images[0], output_path, output_format)

                return GenerationResult(
                    success=True,
                    output=f"Image generated successfully at {output_path}",
                    images=images,
                    image_paths=[str(output_path)],
                    data={
                        "model": model,
                        "aspect_ratio": aspect_ratio,
                        "temperature": temperature,
                        "image_size": image_size,
                        "person_generation": person_generation,
                    },
                )
            else:
                return GenerationResult(
                    success=False,
                    output="",
                    error="No images generated",
                )

        except Exception as e:
            logger.error(f"SDK image generation failed: {e}", exc_info=True)
            return GenerationResult(
                success=False,
                output="",
                error=str(e),
            )

    async def generate(
        self,
        prompt: str,
        output_path: Path,
        aspect_ratio: Optional[str] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        safety_setting: Optional[str] = None,
        image_size: Optional[str] = None,
        person_generation: Optional[str] = None,
        output_format: str = "png",
    ) -> GenerationResult:
        """Generate an image from a text prompt.

        Routes to direct SDK when extended params (image_size,
        person_generation) are specified. Falls back to gemini-imagen
        wrapper for basic calls.
        """
        if self._needs_direct_sdk(image_size, person_generation):
            return await self._generate_via_sdk(
                prompt=prompt,
                output_path=output_path,
                model=model or self.model_name,
                aspect_ratio=aspect_ratio,
                temperature=temperature,
                safety_setting=safety_setting,
                image_size=image_size,
                person_generation=person_generation,
                output_format=output_format,
            )

        try:
            generator = self._get_generator()
            if model:
                generator.model_name = model

            result = await generator.generate(
                prompt=prompt,
                output_images=[str(output_path)],
                aspect_ratio=aspect_ratio,
                temperature=temperature,
            )

            if result.images:
                # Re-save in requested format if not the default png
                if output_format != "png" and output_path.exists():
                    img = Image.open(str(output_path))
                    self._save_image(img, output_path, output_format)

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
        model: Optional[str] = None,
        output_format: str = "png",
    ) -> GenerationResult:
        """Edit an existing image with a text prompt."""
        try:
            generator = self._get_generator()
            if model:
                generator.model_name = model

            result = await generator.generate(
                prompt=prompt,
                input_images=[str(input_path)],
                output_images=[str(output_path)],
                temperature=temperature,
            )

            if result.images:
                if output_format != "png" and output_path.exists():
                    img = Image.open(str(output_path))
                    self._save_image(img, output_path, output_format)

                return GenerationResult(
                    success=True,
                    output=f"Image edited successfully at {output_path}",
                    images=result.images,
                    image_paths=[str(output_path)],
                    data={
                        "model": model or self.model_name,
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
        """Analyze an image and get a description."""
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
        """List available generation models with detailed metadata."""
        models = [
            {
                "name": "gemini-2.5-flash-image",
                "nickname": "Nano Banana",
                "speed": "fast",
                "quality": "good",
                "default": True,
                "description": "Optimized for speed and efficiency. "
                "Best for high-volume, low-latency tasks.",
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
                    "person_generation": True,
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
                    "person_generation": True,
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
                    "person_generation": True,
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
