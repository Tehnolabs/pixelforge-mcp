"""API client for Google Gemini image generation via google-genai SDK."""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_ANALYSIS_PROMPT = (
    "Describe this image in detail, including objects, "
    "colors, composition, and mood."
)

OCR_PROMPT = (
    "Extract all visible text from this image. "
    "Return the result as a JSON object with these fields:\n"
    '- "text": the full extracted text as a single string\n'
    '- "blocks": a list of text blocks, each with "text" and '
    '"confidence" (high/medium/low)\n'
    "Return ONLY valid JSON, no markdown."
)

DETECT_OBJECTS_PROMPT_TEMPLATE = (
    "Detect {target} in this image. For each object found, return "
    "a JSON array where each element has:\n"
    '- "label": the object class name\n'
    '- "box_2d": [y_min, x_min, y_max, x_max] normalized to '
    "0-1000 scale\n"
    '- "confidence": "high", "medium", or "low"\n'
    "Return ONLY valid JSON, no markdown."
)

OPTIMIZE_PROMPT_TEMPLATE = (
    "You are an expert at writing prompts for AI image generation. "
    "Enhance the following prompt to produce better, more detailed "
    "images. Add specific details about:\n"
    "- Lighting and atmosphere\n"
    "- Composition and framing\n"
    "- Colors and textures\n"
    "- Style and artistic technique\n"
    "- Camera angle or perspective\n\n"
    "{style_instruction}"
    "Original prompt: {prompt}\n\n"
    "Return ONLY the enhanced prompt text, nothing else. "
    "Keep it under 500 characters."
)

COMPARE_IMAGES_PROMPT = (
    "Compare these images in detail. Analyze:\n"
    "- Visual differences and similarities\n"
    "- Color, composition, and style differences\n"
    "- Quality and resolution differences\n"
    "- Content and subject matter comparison\n"
    "Provide a structured analysis."
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

# Text-only model for prompt optimization (not image model)
TEXT_MODEL = "gemini-2.5-flash"


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
    """Client for image generation via google-genai SDK.

    All methods use the google-genai SDK directly for generate_content
    (Gemini models) and generate_images (Imagen 4 family).
    """

    # Imagen 4 model name prefixes for auto-routing
    IMAGEN4_MODELS = {
        "imagen-4.0-generate-001",
        "imagen-4.0-ultra-generate-001",
        "imagen-4.0-fast-generate-001",
    }

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-image",
        api_key: Optional[str] = None,
        log_images: bool = False,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.log_images = log_images
        self._genai_client: Optional[Any] = None

    def _get_genai_client(self) -> Any:
        """Get or create direct google-genai client."""
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
                    "No API key found. " "Set GOOGLE_API_KEY environment variable."
                )
            self._genai_client = genai.Client(api_key=api_key)
        return self._genai_client

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
    def _check_safety_block(response: Any) -> Optional["GenerationResult"]:
        """Check response for safety blocks; return a failed result or None."""
        try:
            feedback = response.prompt_feedback
            # Only act when the SDK explicitly set prompt_feedback
            if feedback is None:
                return None
            block_reason = getattr(feedback, "block_reason", None)
            if block_reason is None:
                return None
            # Ensure block_reason is a real value (string or enum), not a
            # Mock auto-attribute.  Real block reasons have a string
            # representation that doesn't start with "<Mock".
            reason_str = str(block_reason)
            if reason_str.startswith("<Mock"):
                return None
            return GenerationResult(
                success=False,
                output="",
                error=f"Content blocked by safety filter: {reason_str}",
            )
        except AttributeError:
            return None

    @staticmethod
    def _classify_error(e: Exception) -> str:
        """Return a user-friendly error message for common API errors."""
        error_type = type(e).__name__
        if error_type == "ResourceExhausted":
            return "Rate limit exceeded. Please wait and retry."
        elif error_type == "PermissionDenied":
            return "API access denied. Check your API key."
        elif error_type == "InvalidArgument":
            return f"Invalid request: {e}"
        return str(e)

    @staticmethod
    def _embed_metadata(
        image_path: Path,
        output_format: str,
        metadata: dict,
    ) -> None:
        """Embed generation metadata into the saved image."""
        try:
            if output_format == "png":
                from PIL.PngImagePlugin import PngInfo

                img = Image.open(str(image_path))
                png_info = PngInfo()
                for key, value in metadata.items():
                    if value is not None:
                        png_info.add_text(f"pixelforge:{key}", str(value))
                img.save(str(image_path), pnginfo=png_info)
            elif output_format in ("jpeg", "webp"):
                # EXIF metadata for JPEG/WebP
                img = Image.open(str(image_path))
                exif = img.getexif()
                # Use ImageDescription (tag 270) for prompt
                if "prompt" in metadata:
                    exif[270] = metadata["prompt"]
                # Use Software (tag 305) for model info
                if "model" in metadata:
                    exif[305] = f"PixelForge MCP ({metadata['model']})"
                img.save(str(image_path), exif=exif)
        except Exception:
            pass  # Don't fail the generation if metadata embedding fails

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
                bg = Image.new("RGB", image.size, (255, 255, 255))
                bg.paste(image, mask=image.split()[-1])
                image = bg
            save_kwargs["quality"] = 95

        image.save(str(output_path), **save_kwargs)

    async def _call_text_model(self, prompt: str, model: str = TEXT_MODEL) -> str:
        """Call a Gemini text model and return the text response."""
        from google.genai import types

        client = self._get_genai_client()
        config = types.GenerateContentConfig(
            response_modalities=["TEXT"],
            temperature=0.7,
        )
        response = await client.aio.models.generate_content(
            model=model, contents=prompt, config=config
        )

        # Check for safety blocks
        blocked = self._check_safety_block(response)
        if blocked is not None:
            raise RuntimeError(blocked.error)

        return self._extract_text_from_response(response)

    async def _analyze_with_images(
        self,
        image_paths: List[Path],
        prompt: str,
        model: Optional[str] = None,
    ) -> GenerationResult:
        """Analyze one or more images with a text prompt via SDK."""
        try:
            from google.genai import types

            client = self._get_genai_client()

            # Build content: images + prompt
            content: List[Any] = []
            for img_path in image_paths:
                img = Image.open(str(img_path))
                content.append(img)
            content.append(prompt)

            config = types.GenerateContentConfig(
                response_modalities=["TEXT"],
            )
            response = await client.aio.models.generate_content(
                model=model or TEXT_MODEL, contents=content, config=config
            )

            # Check for safety blocks
            blocked = self._check_safety_block(response)
            if blocked is not None:
                return blocked

            text = self._extract_text_from_response(response)

            if text:
                return GenerationResult(
                    success=True,
                    output=text,
                    data={"analysis": text},
                )
            else:
                return GenerationResult(
                    success=False,
                    output="",
                    error="No analysis generated",
                )
        except Exception as e:
            error_msg = self._classify_error(e)
            logger.error(f"Analysis failed: {error_msg}", exc_info=True)
            return GenerationResult(success=False, output="", error=error_msg)

    @staticmethod
    def _extract_text_from_response(response: Any) -> str:
        """Extract text from a genai response, handling empty candidates."""
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text or ""
        return ""

    @staticmethod
    def _parse_json_from_text(text: str) -> Any:
        """Extract and parse JSON from model text output."""
        cleaned = text.strip()

        # Find the outermost JSON structure (earliest opening bracket)
        candidates = []
        for open_ch, close_ch in [("{", "}"), ("[", "]")]:
            start = cleaned.find(open_ch)
            end = cleaned.rfind(close_ch)
            if start != -1 and end > start:
                candidates.append((start, cleaned[start : end + 1]))

        # Try candidates in order of earliest start position
        candidates.sort(key=lambda c: c[0])
        for _, fragment in candidates:
            try:
                return json.loads(fragment)
            except json.JSONDecodeError:
                continue

        return None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

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
        reference_images: Optional[List[str]] = None,
        output_format: str = "png",
    ) -> GenerationResult:
        """Generate using google-genai SDK directly."""
        try:
            from google.genai import types

            from .validation import PERSON_GENERATION_SDK_MAP

            client = self._get_genai_client()

            # Build ImageConfig
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

            # Build content with reference images
            contents: List[Any] = []
            if reference_images:
                for ref_path in reference_images:
                    ref_img = Image.open(ref_path)
                    contents.append(ref_img)
            contents.append(prompt)

            # Call Gemini API
            response = await client.aio.models.generate_content(
                model=model, contents=contents, config=config
            )

            # Check for safety blocks
            blocked = self._check_safety_block(response)
            if blocked is not None:
                return blocked

            # Extract image from response
            images: List[Image.Image] = []
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "inline_data") and part.inline_data:
                        img = Image.open(BytesIO(part.inline_data.data))
                        images.append(img)

            if images:
                self._save_image(images[0], output_path, output_format)
                self._embed_metadata(
                    output_path,
                    output_format or "png",
                    {
                        "prompt": prompt,
                        "model": model or self.model_name,
                        "aspect_ratio": aspect_ratio,
                        "temperature": temperature,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
                return GenerationResult(
                    success=True,
                    output=(f"Image generated successfully at {output_path}"),
                    images=images,
                    image_paths=[str(output_path)],
                    data={
                        "model": model,
                        "aspect_ratio": aspect_ratio,
                        "temperature": temperature,
                        "image_size": image_size,
                        "person_generation": person_generation,
                        "reference_images_count": (
                            len(reference_images) if reference_images else 0
                        ),
                    },
                )
            else:
                return GenerationResult(
                    success=False,
                    output="",
                    error="No images generated",
                )

        except Exception as e:
            error_msg = self._classify_error(e)
            logger.error(f"SDK image generation failed: {error_msg}", exc_info=True)
            return GenerationResult(success=False, output="", error=error_msg)

    async def _generate_via_imagen4(
        self,
        prompt: str,
        output_path: Path,
        model: str,
        aspect_ratio: Optional[str] = None,
        image_size: Optional[str] = None,
        person_generation: Optional[str] = None,
        output_format: str = "png",
    ) -> GenerationResult:
        """Generate using Imagen 4 via client.models.generate_images().

        This API is sync-only, so we wrap it with asyncio.to_thread.
        Note: negative_prompt is not supported in Imagen 4.
        """
        try:
            from google.genai import types

            from .validation import PERSON_GENERATION_SDK_MAP

            client = self._get_genai_client()

            config_params: Dict[str, Any] = {
                "number_of_images": 1,
            }
            if aspect_ratio:
                config_params["aspect_ratio"] = aspect_ratio
            if image_size:
                config_params["image_size"] = image_size
            if person_generation:
                sdk_value = PERSON_GENERATION_SDK_MAP.get(person_generation)
                if sdk_value:
                    config_params["person_generation"] = sdk_value

            # Map output_format to MIME type
            mime_map = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}
            config_params["output_mime_type"] = mime_map.get(output_format, "image/png")

            config = types.GenerateImagesConfig(**config_params)

            # generate_images is sync-only — wrap for async compatibility
            response = await asyncio.to_thread(
                client.models.generate_images,
                model=model,
                prompt=prompt,
                config=config,
            )

            if response.generated_images:
                img = response.generated_images[0].image
                self._save_image(img, output_path, output_format)
                self._embed_metadata(
                    output_path,
                    output_format,
                    {
                        "prompt": prompt,
                        "model": model,
                        "aspect_ratio": aspect_ratio,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
                return GenerationResult(
                    success=True,
                    output=f"Image generated successfully at {output_path}",
                    images=[img],
                    image_paths=[str(output_path)],
                    data={
                        "model": model,
                        "aspect_ratio": aspect_ratio,
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
            error_msg = self._classify_error(e)
            logger.error(f"Imagen 4 generation failed: {error_msg}", exc_info=True)
            return GenerationResult(success=False, output="", error=error_msg)

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
        reference_images: Optional[List[str]] = None,
        output_format: str = "png",
    ) -> GenerationResult:
        """Generate an image from a text prompt.

        Auto-routes to the correct SDK method based on model:
        - Imagen 4 models → client.models.generate_images()
        - Gemini models → client.aio.models.generate_content()
        """
        use_model = model or self.model_name

        # Auto-route: Imagen 4 uses a different API
        if use_model in self.IMAGEN4_MODELS:
            return await self._generate_via_imagen4(
                prompt=prompt,
                output_path=output_path,
                model=use_model,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                person_generation=person_generation,
                output_format=output_format,
            )

        return await self._generate_via_sdk(
            prompt=prompt,
            output_path=output_path,
            model=use_model,
            aspect_ratio=aspect_ratio,
            temperature=temperature,
            safety_setting=safety_setting,
            image_size=image_size,
            person_generation=person_generation,
            reference_images=reference_images,
            output_format=output_format,
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
        """Edit an existing image with a text prompt via direct SDK."""
        try:
            from google.genai import types

            client = self._get_genai_client()
            use_model = model or self.model_name

            # Load input image
            input_img = Image.open(str(input_path))

            # Build config
            config_params: Dict[str, Any] = {
                "response_modalities": ["IMAGE"],
            }
            if temperature is not None:
                config_params["temperature"] = temperature
            config = types.GenerateContentConfig(**config_params)

            # Call Gemini API with image + edit prompt
            response = await client.aio.models.generate_content(
                model=use_model,
                contents=[input_img, prompt],
                config=config,
            )

            # Check for safety blocks
            blocked = self._check_safety_block(response)
            if blocked is not None:
                return blocked

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
                    output=f"Image edited successfully at {output_path}",
                    images=images,
                    image_paths=[str(output_path)],
                    data={
                        "model": use_model,
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
            error_msg = self._classify_error(e)
            logger.error(f"Image editing failed: {error_msg}", exc_info=True)
            return GenerationResult(success=False, output="", error=error_msg)

    # ------------------------------------------------------------------
    # Analysis tools
    # ------------------------------------------------------------------

    async def analyze(
        self,
        image_path: Path,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> GenerationResult:
        """Analyze an image and get a description via direct SDK."""
        kwargs: Dict[str, Any] = {}
        if model is not None:
            kwargs["model"] = model
        result = await self._analyze_with_images(
            [image_path], prompt or DEFAULT_ANALYSIS_PROMPT, **kwargs
        )
        # Add image_path to result data for backward compatibility
        if result.success and result.data:
            result.data["image_path"] = str(image_path)
        return result

    async def extract_text(
        self, image_path: Path, model: Optional[str] = None
    ) -> GenerationResult:
        """Extract text from an image using OCR."""
        try:
            kwargs: Dict[str, Any] = {}
            if model is not None:
                kwargs["model"] = model
            text = await self._call_text_model_with_image(
                image_path, OCR_PROMPT, **kwargs
            )

            parsed = self._parse_json_from_text(text)
            if parsed:
                return GenerationResult(
                    success=True,
                    output=parsed.get("text", text),
                    data={
                        "extracted_text": parsed.get("text", ""),
                        "blocks": parsed.get("blocks", []),
                        "raw_response": text,
                        "image_path": str(image_path),
                    },
                )
            else:
                # Fallback: return raw text if JSON parsing fails
                return GenerationResult(
                    success=True,
                    output=text,
                    data={
                        "extracted_text": text,
                        "blocks": [],
                        "raw_response": text,
                        "image_path": str(image_path),
                    },
                )

        except Exception as e:
            error_msg = self._classify_error(e)
            logger.error(f"Text extraction failed: {error_msg}", exc_info=True)
            return GenerationResult(success=False, output="", error=error_msg)

    async def detect_objects(
        self,
        image_path: Path,
        objects: Optional[str] = None,
        model: Optional[str] = None,
    ) -> GenerationResult:
        """Detect objects in an image with bounding boxes."""
        try:
            target = objects or "all visible objects"
            prompt = DETECT_OBJECTS_PROMPT_TEMPLATE.format(target=target)

            det_kwargs: Dict[str, Any] = {}
            if model is not None:
                det_kwargs["model"] = model
            text = await self._call_text_model_with_image(
                image_path, prompt, **det_kwargs
            )

            parsed = self._parse_json_from_text(text)
            if parsed:
                detections = parsed if isinstance(parsed, list) else [parsed]
                return GenerationResult(
                    success=True,
                    output=f"Detected {len(detections)} object(s)",
                    data={
                        "detections": detections,
                        "count": len(detections),
                        "image_path": str(image_path),
                    },
                )
            else:
                return GenerationResult(
                    success=True,
                    output=text,
                    data={
                        "detections": [],
                        "count": 0,
                        "raw_response": text,
                        "image_path": str(image_path),
                    },
                )

        except Exception as e:
            error_msg = self._classify_error(e)
            logger.error(f"Object detection failed: {error_msg}", exc_info=True)
            return GenerationResult(success=False, output="", error=error_msg)

    async def compare_images(
        self,
        image_paths: List[Path],
        prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> GenerationResult:
        """Compare multiple images."""
        kwargs: Dict[str, Any] = {}
        if model is not None:
            kwargs["model"] = model
        return await self._analyze_with_images(
            image_paths, prompt or COMPARE_IMAGES_PROMPT, **kwargs
        )

    # ------------------------------------------------------------------
    # Utility tools
    # ------------------------------------------------------------------

    async def optimize_prompt(
        self,
        prompt: str,
        style: Optional[str] = None,
        model: Optional[str] = None,
    ) -> GenerationResult:
        """Optimize a prompt for better image generation results."""
        try:
            style_instruction = ""
            if style:
                style_instruction = (
                    f"Target style: {style}. Tailor the prompt "
                    f"specifically for {style} output.\n\n"
                )

            meta_prompt = OPTIMIZE_PROMPT_TEMPLATE.format(
                prompt=prompt, style_instruction=style_instruction
            )

            opt_kwargs: Dict[str, Any] = {}
            if model is not None:
                opt_kwargs["model"] = model
            enhanced = await self._call_text_model(meta_prompt, **opt_kwargs)
            enhanced = enhanced.strip().strip('"')

            if enhanced:
                return GenerationResult(
                    success=True,
                    output=enhanced,
                    data={
                        "original_prompt": prompt,
                        "enhanced_prompt": enhanced,
                        "style": style,
                    },
                )
            else:
                return GenerationResult(
                    success=False,
                    output="",
                    error="Prompt optimization failed",
                )

        except Exception as e:
            error_msg = self._classify_error(e)
            logger.error(f"Prompt optimization failed: {error_msg}", exc_info=True)
            return GenerationResult(success=False, output="", error=error_msg)

    async def remove_background(
        self,
        image_path: Path,
        output_path: Path,
        output_format: str = "png",
    ) -> GenerationResult:
        """Remove background from an image via direct SDK.

        Delegates to edit() with a fixed background-removal prompt.
        """
        result = await self.edit(
            prompt=(
                "Remove the background completely. Keep only the "
                "main subject. Make the background transparent "
                "or pure white."
            ),
            input_path=image_path,
            output_path=output_path,
            output_format=output_format,
        )
        # Adjust response for remove_background semantics
        if result.success:
            result.output = f"Background removed. Saved to {output_path}"
            result.data = {"image_path": str(output_path)}
        elif result.error == "No images generated":
            result.error = "Background removal failed"
        return result

    async def _call_text_model_with_image(
        self,
        image_path: Path,
        prompt: str,
        model: Optional[str] = None,
    ) -> str:
        """Call text model with a single image + prompt."""
        from google.genai import types

        client = self._get_genai_client()
        img = Image.open(str(image_path))

        config = types.GenerateContentConfig(
            response_modalities=["TEXT"],
        )
        response = await client.aio.models.generate_content(
            model=model or TEXT_MODEL,
            contents=[img, prompt],
            config=config,
        )

        # Check for safety blocks
        blocked = self._check_safety_block(response)
        if blocked is not None:
            raise RuntimeError(blocked.error)

        return self._extract_text_from_response(response)

    # ------------------------------------------------------------------
    # Info tools
    # ------------------------------------------------------------------

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
                    "reference_inputs": ("up to 14 objects + 5 characters"),
                    "thinking_process": True,
                    "grounding": ("web search + Google Image Search"),
                    "person_generation": True,
                },
                "temperature": {
                    "min": 0.0,
                    "max": 2.0,
                    "default": 1.0,
                },
            },
            {
                "name": "imagen-4.0-generate-001",
                "nickname": "Imagen 4",
                "speed": "moderate",
                "quality": "excellent",
                "default": False,
                "description": "Dedicated image generation model. "
                "Cheapest at ~$0.04/img. Text-to-image only.",
                "best_for": [
                    "Cost-effective batch generation",
                    "High-quality standalone images",
                    "When editing is not needed",
                ],
                "capabilities": {
                    "text_rendering": "good",
                    "complex_scenes": "excellent",
                    "editing": False,
                    "resolution": "1K/2K",
                    "person_generation": True,
                },
                "temperature": None,
                "note": "Uses a separate API (generate_images). "
                "No editing, no temperature control.",
            },
            {
                "name": "imagen-4.0-fast-generate-001",
                "nickname": "Imagen 4 Fast",
                "speed": "fast",
                "quality": "good",
                "default": False,
                "description": "Fastest Imagen model. ~$0.02/img.",
                "best_for": [
                    "Rapid prototyping",
                    "High-volume generation",
                    "Cost-sensitive workloads",
                ],
                "capabilities": {
                    "text_rendering": "basic",
                    "complex_scenes": "good",
                    "editing": False,
                    "resolution": "1K",
                    "person_generation": True,
                },
                "temperature": None,
                "note": "Uses a separate API (generate_images). "
                "No editing, no temperature control.",
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
                    "quality+speed and panoramic ratios, "
                    "imagen-4.0-fast-generate-001 for cheapest"
                ),
            },
        )
