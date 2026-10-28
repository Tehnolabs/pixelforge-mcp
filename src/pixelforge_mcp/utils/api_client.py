"""API client for Google Gemini image generation via google-genai SDK."""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from PIL import Image

from .validation import (
    DEFAULT_ANALYSIS_MODEL,
    DEFAULT_MODEL,
    DEPRECATED_MODEL_ALIASES,
    RETIRED_MODELS,
)

if TYPE_CHECKING:
    from google.genai import types

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

# Default text model for analysis and prompt optimization (not an image model)
TEXT_MODEL = DEFAULT_ANALYSIS_MODEL


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

    All methods use the google-genai SDK directly via generate_content
    (Gemini image and text models).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
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
    def _retired_model_error(model: str) -> Optional["GenerationResult"]:
        """Return a failed result if the model was retired, else None."""
        replacement = RETIRED_MODELS.get(model)
        if replacement is None:
            return None
        return GenerationResult(
            success=False,
            output="",
            error=(
                f"Model '{model}' was retired by Google on 2026-08-17 and is "
                "no longer served by the Gemini API. "
                f"Use '{replacement}' instead (quality='balanced')."
            ),
        )

    @staticmethod
    def _resolve_deprecated_alias(model: str) -> Tuple[str, Optional[str]]:
        """Map a shut-down preview ID to its stable replacement.

        Returns the model to call and a warning when it was rewritten.
        """
        replacement = DEPRECATED_MODEL_ALIASES.get(model)
        if replacement is None:
            return model, None
        return replacement, (
            f"Model '{model}' was retired by Google on 2026-06-25. "
            f"Using '{replacement}' instead. "
            f"Update your configuration to use '{replacement}' directly."
        )

    @staticmethod
    def _build_safety_settings(
        safety_setting: Optional[str],
    ) -> Optional[List["types.SafetySetting"]]:
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
            feedback = getattr(response, "prompt_feedback", None)
            if feedback is None:
                return None
            block_reason = getattr(feedback, "block_reason", None)
            if block_reason is None:
                return None
            return GenerationResult(
                success=False,
                output="",
                error=f"Content blocked by safety filter: {block_reason}",
            )
        except Exception:
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
        metadata: Dict[str, Any],
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
        use_grounding: bool = False,
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

            config_params: Dict[str, Any] = {
                "response_modalities": ["TEXT"],
            }
            if use_grounding:
                config_params["tools"] = [
                    types.Tool(google_search=types.GoogleSearch())
                ]

            config = types.GenerateContentConfig(**config_params)
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
        thinking_budget: Optional[int] = None,
        alias_warning: Optional[str] = None,
    ) -> GenerationResult:
        """Generate using google-genai SDK directly."""
        try:
            from google.genai import types

            client = self._get_genai_client()

            # Build ImageConfig
            image_config_params: Dict[str, Any] = {}
            if aspect_ratio:
                image_config_params["aspect_ratio"] = aspect_ratio
            if image_size:
                image_config_params["image_size"] = image_size
            # NOTE: person_generation is NOT passed to ImageConfig.
            # The Gemini API (generate_content) does not support it —
            # the SDK raises ValueError in _ImageConfig_to_mldev().

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

            # Thinking config (for models that support it)
            if thinking_budget is not None and thinking_budget > 0:
                config_params["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=thinking_budget
                )

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
                data: Dict[str, Any] = {
                    "model": model,
                    "aspect_ratio": aspect_ratio,
                    "temperature": temperature,
                    "image_size": image_size,
                    "person_generation": None,
                    "reference_images_count": (
                        len(reference_images) if reference_images else 0
                    ),
                }
                warnings: List[str] = []
                if alias_warning:
                    warnings.append(alias_warning)
                if person_generation:
                    warnings.append(
                        f"person_generation='{person_generation}' is accepted"
                        " for compatibility but is not enforced by the Gemini"
                        " API. Gemini applies built-in restrictions on"
                        " photorealistic identifiable people."
                    )
                if warnings:
                    data["warnings"] = warnings
                return GenerationResult(
                    success=True,
                    output=(f"Image generated successfully at {output_path}"),
                    images=images,
                    image_paths=[str(output_path)],
                    data=data,
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
        thinking_budget: Optional[int] = None,
    ) -> GenerationResult:
        """Generate an image from a text prompt.

        Calls client.aio.models.generate_content(). Retired model IDs
        (see RETIRED_MODELS) return an error naming the replacement.
        Shut-down preview IDs are rewritten to their stable replacements
        with a warning.
        """
        use_model = model or self.model_name

        retired = self._retired_model_error(use_model)
        if retired is not None:
            return retired
        use_model, alias_warning = self._resolve_deprecated_alias(use_model)

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
            thinking_budget=thinking_budget,
            alias_warning=alias_warning,
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
        use_model = model or self.model_name

        retired = self._retired_model_error(use_model)
        if retired is not None:
            return retired
        use_model, alias_warning = self._resolve_deprecated_alias(use_model)

        try:
            from google.genai import types

            client = self._get_genai_client()

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
                edit_data: Dict[str, Any] = {
                    "model": use_model,
                    "temperature": temperature,
                }
                if alias_warning:
                    edit_data["warnings"] = [alias_warning]
                return GenerationResult(
                    success=True,
                    output=f"Image edited successfully at {output_path}",
                    images=images,
                    image_paths=[str(output_path)],
                    data=edit_data,
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
        use_grounding: bool = False,
    ) -> GenerationResult:
        """Analyze an image and get a description via direct SDK."""
        kwargs: Dict[str, Any] = {}
        if model is not None:
            kwargs["model"] = model
        if use_grounding:
            kwargs["use_grounding"] = True
        result = await self._analyze_with_images(
            [image_path],
            prompt or DEFAULT_ANALYSIS_PROMPT,
            **kwargs,
        )
        # Add image_path to result data for backward compatibility
        if result.success and result.data:
            result.data["image_path"] = str(image_path)
        return result

    async def extract_text(
        self,
        image_path: Path,
        model: Optional[str] = None,
        use_grounding: bool = False,
    ) -> GenerationResult:
        """Extract text from an image using OCR."""
        try:
            kwargs: Dict[str, Any] = {}
            if model is not None:
                kwargs["model"] = model
            if use_grounding:
                kwargs["use_grounding"] = True
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
        use_grounding: bool = False,
    ) -> GenerationResult:
        """Detect objects in an image with bounding boxes."""
        try:
            target = objects or "all visible objects"
            prompt = DETECT_OBJECTS_PROMPT_TEMPLATE.format(target=target)

            det_kwargs: Dict[str, Any] = {}
            if model is not None:
                det_kwargs["model"] = model
            if use_grounding:
                det_kwargs["use_grounding"] = True
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
        use_grounding: bool = False,
    ) -> GenerationResult:
        """Compare multiple images."""
        kwargs: Dict[str, Any] = {}
        if model is not None:
            kwargs["model"] = model
        if use_grounding:
            kwargs["use_grounding"] = True
        return await self._analyze_with_images(
            image_paths,
            prompt or COMPARE_IMAGES_PROMPT,
            **kwargs,
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
        use_grounding: bool = False,
    ) -> str:
        """Call text model with a single image + prompt."""
        from google.genai import types

        client = self._get_genai_client()
        img = Image.open(str(image_path))

        config_params: Dict[str, Any] = {
            "response_modalities": ["TEXT"],
        }
        if use_grounding:
            config_params["tools"] = [types.Tool(google_search=types.GoogleSearch())]

        config = types.GenerateContentConfig(**config_params)
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
    # Vertex AI tools (optional)
    # ------------------------------------------------------------------

    def _get_vertex_client(self) -> Any:
        """Get a Vertex AI-authenticated genai client.

        Raises ImportError if google-cloud-aiplatform is not installed.
        """
        try:
            from google import genai

            # Vertex AI client uses project/location instead of API key
            from pixelforge_mcp.config import get_config

            config = get_config()

            return genai.Client(
                vertexai=True,
                project=config.vertex.project_id,
                location=config.vertex.location,
            )
        except ImportError:
            raise ImportError(
                "Vertex AI requires google-cloud-aiplatform. "
                "Install with: pip install pixelforge-mcp[vertex]"
            )

    async def upscale(
        self,
        image_path: Path,
        output_path: Path,
        upscale_factor: str = "x2",
        output_format: str = "png",
    ) -> GenerationResult:
        """Upscale an image using Vertex AI (requires Vertex credentials)."""
        try:
            from google.genai import types

            client = self._get_vertex_client()

            # The SDK expects types.Image; a PIL image is coerced to one with
            # no bytes, which sends an empty image to the API.
            response = await asyncio.to_thread(
                client.models.upscale_image,
                model="imagen-4.0-upscale-preview",
                image=types.Image.from_file(location=str(image_path)),
                upscale_factor=upscale_factor,
                config=types.UpscaleImageConfig(
                    output_mime_type="image/png",
                ),
            )

            if response.generated_images:
                genai_img = response.generated_images[0].image
                result_img = Image.open(BytesIO(genai_img.image_bytes))
                self._save_image(result_img, output_path, output_format)
                return GenerationResult(
                    success=True,
                    output=f"Image upscaled ({upscale_factor}) to {output_path}",
                    images=[result_img],
                    image_paths=[str(output_path)],
                    data={
                        "upscale_factor": upscale_factor,
                        "image_path": str(output_path),
                    },
                )
            else:
                return GenerationResult(
                    success=False,
                    output="",
                    error="Upscaling produced no output",
                )

        except ImportError as e:
            return GenerationResult(success=False, output="", error=str(e))
        except Exception as e:
            error_msg = self._classify_error(e)
            logger.error(f"Upscaling failed: {error_msg}", exc_info=True)
            return GenerationResult(success=False, output="", error=error_msg)

    async def advanced_edit(
        self,
        image_path: Path,
        prompt: str,
        edit_mode: str,
        output_path: Path,
        mask_path: Optional[Path] = None,
        output_format: str = "png",
    ) -> GenerationResult:
        """Advanced image editing using Vertex AI EditImage API."""
        try:
            from google.genai import types

            client = self._get_vertex_client()

            # Map user-friendly mode names to SDK enum values
            mode_map: Dict[str, types.EditMode] = {
                "inpaint_removal": types.EditMode.EDIT_MODE_INPAINT_REMOVAL,
                "inpaint_insertion": types.EditMode.EDIT_MODE_INPAINT_INSERTION,
                "outpaint": types.EditMode.EDIT_MODE_OUTPAINT,
                "background_swap": types.EditMode.EDIT_MODE_BGSWAP,
                "style_transfer": types.EditMode.EDIT_MODE_STYLE,
                "product_image": types.EditMode.EDIT_MODE_PRODUCT_IMAGE,
            }
            sdk_mode = mode_map.get(edit_mode)
            if not sdk_mode:
                return GenerationResult(
                    success=False,
                    output="",
                    error=f"Unknown edit mode: {edit_mode}",
                )

            # Build reference images (the SDK expects types.Image, not PIL)
            reference_images: List[
                Union[types.RawReferenceImage, types.MaskReferenceImage]
            ] = [
                types.RawReferenceImage(
                    reference_image=types.Image.from_file(location=str(image_path)),
                    reference_id=0,
                )
            ]
            if mask_path:
                reference_images.append(
                    types.MaskReferenceImage(
                        reference_image=types.Image.from_file(location=str(mask_path)),
                        reference_id=1,
                        config=types.MaskReferenceConfig(
                            mask_mode=types.MaskReferenceMode.MASK_MODE_USER_PROVIDED,
                        ),
                    )
                )

            config = types.EditImageConfig(
                edit_mode=sdk_mode,
                number_of_images=1,
                output_mime_type="image/png",
            )

            response = await asyncio.to_thread(
                client.models.edit_image,
                model="imagen-3.0-capability-001",
                prompt=prompt,
                reference_images=reference_images,
                config=config,
            )

            if response.generated_images:
                genai_img = response.generated_images[0].image
                result_img = Image.open(BytesIO(genai_img.image_bytes))
                self._save_image(result_img, output_path, output_format)
                return GenerationResult(
                    success=True,
                    output=f"Advanced edit ({edit_mode}) applied to {output_path}",
                    images=[result_img],
                    image_paths=[str(output_path)],
                    data={
                        "edit_mode": edit_mode,
                        "image_path": str(output_path),
                    },
                )
            else:
                return GenerationResult(
                    success=False,
                    output="",
                    error="Advanced edit produced no output",
                )

        except ImportError as e:
            return GenerationResult(success=False, output="", error=str(e))
        except Exception as e:
            error_msg = self._classify_error(e)
            logger.error(f"Advanced edit failed: {error_msg}", exc_info=True)
            return GenerationResult(success=False, output="", error=error_msg)

    # ------------------------------------------------------------------
    # Info tools
    # ------------------------------------------------------------------

    async def list_models(self) -> GenerationResult:
        """List available generation models with detailed metadata."""
        temperature = {"min": 0.0, "max": 2.0, "default": 1.0}
        models: List[Dict[str, Any]] = [
            {
                "name": "gemini-nano-banana-2.1",
                "nickname": "Nano Banana 2.1",
                "speed": "fast",
                "quality": "excellent",
                "default": True,
                "description": "Google's recommended default. Excellent quality "
                "at Flash speed with grounding, thinking levels, and "
                "panoramic ratios.",
                "best_for": [
                    "General-purpose generation (default)",
                    "Panoramic aspect ratios (1:4, 4:1, 1:8, 8:1)",
                    "Grounded generation (web + Google Image Search)",
                    "Multi-turn editing and up to 14 reference images",
                    "Output up to 4K",
                ],
                "capabilities": {
                    "text_rendering": "excellent",
                    "complex_scenes": "excellent",
                    "editing": "multi-turn",
                    "resolution": "1K/2K/4K",
                    "person_generation": False,
                    "grounding": "web search + Google Image Search",
                    "thinking": "minimal/medium/high",
                    "panoramic": True,
                    "reference_images": 14,
                },
                "temperature": dict(temperature),
                "pricing": {"0.5K": None, "1K": 0.0336, "2K": 0.0504, "4K": 0.113},
            },
            {
                "name": "gemini-3.1-flash-lite-image",
                "nickname": "Nano Banana 2 Lite",
                "speed": "fastest",
                "quality": "good",
                "default": False,
                "description": "Fastest and cheapest stable model. 1K output "
                "only, no panoramic ratios, no grounding. Not optimized for "
                "multiple reference inputs or multi-turn editing.",
                "best_for": [
                    "Rapid iteration and prototyping",
                    "High-volume, cost-sensitive generation",
                    "Standard aspect ratios at 1K",
                ],
                "capabilities": {
                    "text_rendering": "good",
                    "complex_scenes": "moderate",
                    "editing": "basic (not optimized for multi-turn)",
                    "resolution": "1K",
                    "person_generation": False,
                    "grounding": False,
                    "thinking": None,
                    "panoramic": False,
                    "reference_images": None,
                },
                "temperature": dict(temperature),
                "pricing": {"0.5K": None, "1K": 0.0336, "2K": None, "4K": None},
            },
            {
                "name": "gemini-3.1-flash-image",
                "nickname": "Nano Banana 2",
                "speed": "fast",
                "quality": "excellent",
                "default": False,
                "description": "Previous-generation workhorse. Panoramic "
                "ratios, grounding, and a 0.5K tier for rapid iteration.",
                "best_for": [
                    "Panoramic aspect ratios (1:4, 4:1, 1:8, 8:1)",
                    "Grounded generation (web + Google Image Search)",
                    "0.5K rapid iteration tier",
                    "Up to 14 reference images",
                ],
                "capabilities": {
                    "text_rendering": "excellent",
                    "complex_scenes": "excellent",
                    "editing": "multi-turn",
                    "resolution": "0.5K/1K/2K/4K",
                    "person_generation": False,
                    "grounding": "web search + Google Image Search",
                    "thinking": True,
                    "panoramic": True,
                    "reference_images": 14,
                },
                "temperature": dict(temperature),
                "pricing": {"0.5K": 0.045, "1K": 0.067, "2K": 0.101, "4K": 0.151},
            },
            {
                "name": "gemini-3-pro-image",
                "nickname": "Nano Banana Pro",
                "speed": "moderate",
                "quality": "excellent",
                "default": False,
                "description": "Highest quality and best text rendering, "
                "with deep reasoning for complex prompts.",
                "best_for": [
                    "Highest text rendering accuracy",
                    "Complex multi-turn editing workflows",
                    "Deep reasoning on intricate prompts",
                    "Character and style consistency",
                ],
                "capabilities": {
                    "text_rendering": "excellent",
                    "complex_scenes": "excellent",
                    "editing": "advanced multi-turn",
                    "resolution": "1K/2K/4K",
                    "person_generation": False,
                    "grounding": "web search",
                    "thinking": "always on",
                    "panoramic": False,
                    "reference_images": 6,
                },
                "temperature": dict(temperature),
                "pricing": {"0.5K": None, "1K": 0.134, "2K": 0.134, "4K": 0.24},
            },
            {
                "name": "gemini-2.5-flash-image",
                "nickname": "Nano Banana",
                "speed": "fast",
                "quality": "good",
                "default": False,
                "description": "Deprecated legacy model. Google has announced "
                "its shutdown.",
                "best_for": [
                    "Existing workflows that have not migrated yet",
                ],
                "capabilities": {
                    "text_rendering": "basic",
                    "complex_scenes": "moderate",
                    "editing": "basic",
                    "resolution": "1K",
                    "person_generation": False,
                    "grounding": False,
                    "thinking": False,
                    "panoramic": False,
                    "reference_images": 3,
                },
                "temperature": dict(temperature),
                "pricing": {"0.5K": None, "1K": 0.039, "2K": None, "4K": None},
                "deprecated": True,
                "replacement": "gemini-3.1-flash-lite-image",
            },
        ]

        analysis_models: List[Dict[str, Any]] = [
            {
                "name": "gemini-3.5-flash-lite",
                "nickname": "Gemini 3.5 Flash-Lite",
                "default": True,
                "description": "Default for analyze_image, extract_text, "
                "detect_objects, compare_images, and optimize_prompt.",
                "best_for": [
                    "Everyday image description and prompt optimization",
                    "Low-cost, low-latency analysis",
                ],
                "pricing_per_1m_tokens": {"input": 0.30, "output": 2.50},
            },
            {
                "name": "gemini-3.8-flash",
                "nickname": "Gemini 3.8 Flash",
                "default": False,
                "description": "Strongest analysis model. Pass "
                "model='gemini-3.8-flash' to an analysis tool.",
                "best_for": [
                    "Hardest OCR and reasoning over images",
                    "Precise object detection",
                ],
                "pricing_per_1m_tokens": {"input": 0.75, "output": 3.75},
            },
        ]

        return GenerationResult(
            success=True,
            output=f"Found {len(models)} available models",
            data={
                "models": models,
                "analysis_models": analysis_models,
                "recommendation": (
                    "Use quality presets: fast (gemini-3.1-flash-lite-image), "
                    "balanced (gemini-nano-banana-2.1, default), "
                    "quality (gemini-3-pro-image). "
                    "Analysis tools default to gemini-3.5-flash-lite; pass "
                    "model='gemini-3.8-flash' for harder OCR or reasoning."
                ),
            },
        )
