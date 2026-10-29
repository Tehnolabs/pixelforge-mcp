"""API client for Google Gemini image generation via google-genai SDK."""

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from PIL import Image, UnidentifiedImageError

from pixelforge_mcp.config import prefer_vertex_from_env

from .validation import (
    DEFAULT_ANALYSIS_MODEL,
    DEFAULT_MODEL,
    DEPRECATED_MODEL_ALIASES,
    IMAGE_SIZE_PIXELS,
    RETIRED_MODELS,
    SDK_IMAGE_SIZE_VALUES,
    UPSCALE_FACTORS,
    supported_aspect_ratios,
    supported_image_sizes,
)

if TYPE_CHECKING:
    from google.genai import types

logger = logging.getLogger(__name__)

# Backends. The Gemini API is selected by an API key; Vertex AI by a
# Google Cloud project with Application Default Credentials.
BACKEND_GEMINI_API = "gemini_api"
BACKEND_VERTEX_AI = "vertex_ai"

# The current Gemini image and text models are served from this Vertex AI
# location. Regional endpoints return 404 for them.
DEFAULT_VERTEX_LOCATION = "global"

# Retry policy for transient API errors. google-genai only retries when
# retry options are set; image models on a fresh project return 429 on
# short bursts.
RETRY_ATTEMPTS = 5
RETRY_INITIAL_DELAY = 2.0
RETRY_MAX_DELAY = 60.0
RETRY_STATUS_CODES = [408, 429, 500, 502, 503, 504]

# A chosen aspect ratio within this factor of the source shape needs no warning.
ASPECT_RATIO_TOLERANCE = 1.05

UPSCALE_INSTRUCTION = (
    "Upscale this image to a higher resolution. Reproduce it exactly: the "
    "same composition, subjects, colors, lighting, text, and framing. Only "
    "add fine detail and sharpness. Do not add, remove, move, or restyle "
    "anything."
)

MASK_INSTRUCTION = (
    "The second image is a mask for the first image: white areas mark the "
    "region to change, black areas must stay exactly as they are."
)

# Mode-specific instructions for advanced_edit. {prompt} is the user's text.
EDIT_MODE_INSTRUCTIONS: Dict[str, str] = {
    "inpaint_removal": (
        "Remove the following from the image: {prompt}. Fill the vacated "
        "area so it blends seamlessly with the surroundings. Keep "
        "everything else exactly as it is."
    ),
    "inpaint_insertion": (
        "Add the following to the image: {prompt}. Integrate it naturally "
        "with matching lighting, perspective, scale, and shadows. Keep "
        "everything else exactly as it is."
    ),
    "outpaint": (
        "Extend the image beyond its current borders, continuing the scene "
        "naturally and consistently. {prompt}. Keep the original content "
        "unchanged in the center."
    ),
    "background_swap": (
        "Replace the background with: {prompt}. Keep the main subject "
        "exactly as it is, with clean edges and lighting that matches the "
        "new background."
    ),
    "style_transfer": (
        "Re-render the image in this style: {prompt}. Preserve the "
        "composition, subjects, and layout."
    ),
    "product_image": (
        "Create a professional product photograph of the main subject: "
        "{prompt}. Keep the product's shape, colors, labels, and "
        "proportions exactly as shown."
    ),
}

GENERATIVE_UPSCALE_WARNING = (
    "Generative upscale re-renders the image with a Gemini image model, so "
    "fine details can differ from the source. For exact pixel scaling use "
    "transform_image(operation='resize')."
)

MASK_GUIDANCE_WARNING = (
    "The mask guides the model semantically; the edited region is not " "pixel-exact."
)

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
    (Gemini image and text models). Requests go to the Gemini API with an
    API key, or to Vertex AI with a Google Cloud project and Application
    Default Credentials.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        log_images: bool = False,
        backend: Optional[str] = None,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.log_images = log_images
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = (
            location or os.getenv("GOOGLE_CLOUD_LOCATION") or DEFAULT_VERTEX_LOCATION
        )
        self.backend = backend or self._detect_backend()
        self._genai_client: Optional[Any] = None

    def _resolved_api_key(self) -> Optional[str]:
        return (
            self.api_key
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )

    def _detect_backend(self) -> str:
        """Pick the backend from the project, the API key, and SDK switches.

        Vertex AI needs a project. It is chosen when no API key is set, or
        when GOOGLE_GENAI_USE_VERTEXAI / GOOGLE_GENAI_USE_ENTERPRISE is on.
        """
        if self.project_id and (
            prefer_vertex_from_env() or not self._resolved_api_key()
        ):
            return BACKEND_VERTEX_AI
        return BACKEND_GEMINI_API

    @staticmethod
    def _http_options() -> Any:
        """Retry transient errors (rate limits, 5xx) with backoff."""
        from google.genai import types

        return types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                attempts=RETRY_ATTEMPTS,
                initial_delay=RETRY_INITIAL_DELAY,
                max_delay=RETRY_MAX_DELAY,
                exp_base=2.0,
                jitter=1.0,
                http_status_codes=RETRY_STATUS_CODES,
            )
        )

    def _get_genai_client(self) -> Any:
        """Get or create the google-genai client for the active backend."""
        if self._genai_client is None:
            from google import genai

            if self.backend == BACKEND_VERTEX_AI:
                if not self.project_id:
                    raise ValueError(
                        "Vertex AI backend selected but no project is "
                        "configured. Set GOOGLE_CLOUD_PROJECT, and "
                        "GOOGLE_APPLICATION_CREDENTIALS for a service account."
                    )
                self._genai_client = genai.Client(
                    vertexai=True,
                    project=self.project_id,
                    location=self.location,
                    http_options=self._http_options(),
                )
            else:
                api_key = self._resolved_api_key()
                if not api_key:
                    raise ValueError(
                        "No credentials found. Set GOOGLE_API_KEY for the "
                        "Gemini API, or GOOGLE_CLOUD_PROJECT with Application "
                        "Default Credentials for Vertex AI."
                    )
                # vertexai=False pins the Gemini API. Left unset, the SDK
                # reads GOOGLE_GENAI_USE_VERTEXAI / GOOGLE_GENAI_USE_ENTERPRISE
                # and would send the API key to Vertex AI instead.
                self._genai_client = genai.Client(
                    vertexai=False,
                    api_key=api_key,
                    http_options=self._http_options(),
                )
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
        """Return a user-friendly error message for common API errors.

        google-genai raises ClientError/ServerError carrying ``code`` (HTTP
        status), ``status`` (API status name) and ``message`` (the API's
        own text); plain exceptions are matched on their text. The original
        text is kept so a quota or permission problem stays visible.
        """
        code = getattr(e, "code", None)
        status = str(getattr(e, "status", None) or "")
        detail = str(getattr(e, "message", None) or e).strip()
        lower = f"{status} {detail}".lower()

        if code == 429 or "resource_exhausted" in lower:
            return (
                f"Rate limit exceeded after {RETRY_ATTEMPTS} attempts "
                f"({status or 'RESOURCE_EXHAUSTED'}): {detail[:300]} If this "
                "persists or the quota limit is 0, the project needs a quota "
                "increase; retrying will not help."
            )
        if code == 403 or "permission_denied" in lower:
            return (
                f"Permission denied ({status or 'PERMISSION_DENIED'}): "
                f"{detail[:300]} On Vertex AI, enable the Vertex AI API on the "
                "project and grant the identity roles/aiplatform.user. With an "
                "API key, check the key and that the Generative Language API "
                "is enabled."
            )
        if "publisher model" in lower and "not found" in lower:
            if "/locations/global/" in lower:
                return (
                    f"{detail} The model ID is unknown or retired on Vertex AI "
                    "(call list_available_models for current IDs), or the "
                    "project cannot access it yet."
                )
            return (
                f"{detail} The current Gemini image and text models are served "
                "from the 'global' Vertex AI location; set "
                "GOOGLE_CLOUD_LOCATION=global."
            )
        if "api key not valid" in lower or "api_key_invalid" in lower:
            return f"Invalid API key: {detail[:300]}"
        if code == 400 or "invalid_argument" in lower:
            return f"Invalid request ({status or 'INVALID_ARGUMENT'}): {detail[:300]}"
        if code:
            prefix = " ".join(str(part) for part in (code, status) if part)
            return f"{prefix}: {detail}"
        return detail

    @staticmethod
    def _extract_images(response: Any) -> Tuple[List[Image.Image], Optional[str]]:
        """Return the images in a response, or the reason none came back.

        A response can carry no candidates, a candidate without content,
        or content whose parts are None (for example when generation was
        stopped by an image safety filter).
        """
        candidate = response.candidates[0] if response.candidates else None
        content = getattr(candidate, "content", None) if candidate else None
        parts = getattr(content, "parts", None) or []

        images: List[Image.Image] = []
        undecodable = 0
        for part in parts:
            if getattr(part, "inline_data", None) and part.inline_data.data:
                try:
                    images.append(Image.open(BytesIO(part.inline_data.data)))
                except UnidentifiedImageError:
                    undecodable += 1
        if images:
            return images, None

        finish = getattr(candidate, "finish_reason", None) if candidate else None
        finish_name = getattr(finish, "name", None)
        if not isinstance(finish_name, str):
            finish_name = str(finish) if isinstance(finish, str) else "unknown"
        if candidate is None:
            finish_name = "no candidates"
        reason = f"No image returned (finish_reason={finish_name})"
        if undecodable:
            reason += f"; {undecodable} undecodable image part(s)"
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                reason += f": {text.strip()[:200]}"
                break
        return [], reason

    @staticmethod
    def _sdk_image_size(image_size: Optional[str]) -> Optional[str]:
        """Map a tool-facing size (e.g. 0.5K) to the value the API expects."""
        if image_size is None:
            return None
        return SDK_IMAGE_SIZE_VALUES.get(image_size, image_size)

    @staticmethod
    def _ratio_value(ratio: str) -> float:
        w, h = ratio.split(":")
        return int(w) / int(h)

    @staticmethod
    def _nearest_aspect_ratio(
        width: int, height: int, model: Optional[str] = None
    ) -> str:
        """Pick the ratio closest to an image's shape among those the model honours.

        Distances are compared in log space so wide and tall sources are
        treated alike. Models without panoramic support get the nearest
        standard ratio; unknown models may use any ratio.
        """
        if width <= 0 or height <= 0:
            return "1:1"
        target = math.log(width / height)
        best = "1:1"
        best_diff = float("inf")
        for ratio in supported_aspect_ratios(model):
            diff = abs(math.log(ImagenAPIClient._ratio_value(ratio)) - target)
            if diff < best_diff:
                best, best_diff = ratio, diff
        return best

    @staticmethod
    def _aspect_ratio_warning(
        width: int, height: int, ratio: str, model: str
    ) -> Optional[str]:
        """Warn when the chosen ratio differs from the source shape."""
        if width <= 0 or height <= 0:
            return None
        source = math.log(width / height)
        drift = abs(math.log(ImagenAPIClient._ratio_value(ratio)) - source)
        if drift <= math.log(ASPECT_RATIO_TOLERANCE):
            return None
        return (
            f"Source is {width}x{height}; {model} renders at {ratio}, the "
            "nearest ratio it supports, so the output shape differs from the "
            "source."
        )

    @staticmethod
    def _size_tier_for_edge(long_edge: int) -> str:
        """Smallest 1K/2K/4K tier whose longest edge covers ``long_edge``."""
        for tier in ("1K", "2K", "4K"):
            if long_edge <= IMAGE_SIZE_PIXELS[tier]:
                return tier
        return "4K"

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
                image_config_params["image_size"] = self._sdk_image_size(image_size)
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

            images, no_image_reason = self._extract_images(response)

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
                    error=no_image_reason or "No images generated",
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

            images, no_image_reason = self._extract_images(response)

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
                    error=no_image_reason or "No images generated",
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
            warnings = (result.data or {}).get("warnings")
            result.data = {"image_path": str(output_path)}
            if warnings:
                result.data["warnings"] = warnings
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
    # Generative upscale and mode-driven editing
    # ------------------------------------------------------------------
    # Imagen's dedicated upscale and edit models were retired by Google in
    # 2026. Both tools now re-render with a Gemini image model through
    # generate_content on whichever backend is configured.

    @staticmethod
    def _upscale_target(
        width: int, height: int, upscale_factor: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Choose the output size tier for an upscale.

        Returns (image_size, warning, error). The tier is the smallest of
        1K/2K/4K whose longest edge covers ``longest edge * factor``,
        capped at 4K.
        """
        factor = UPSCALE_FACTORS[upscale_factor]
        long_edge = max(width, height)
        wanted = long_edge * factor
        tier = ImagenAPIClient._size_tier_for_edge(wanted)
        tier_px = IMAGE_SIZE_PIXELS[tier]
        if tier_px <= long_edge:
            return (
                None,
                None,
                f"Image is already {long_edge}px on its longest edge; the "
                "maximum output is 4K (4096px).",
            )
        warning = None
        if wanted > tier_px:
            warning = (
                f"{upscale_factor} would need {wanted}px; output is capped "
                f"at {tier} ({tier_px}px)."
            )
        return tier, warning, None

    async def upscale(
        self,
        image_path: Path,
        output_path: Path,
        upscale_factor: str = "x2",
        output_format: str = "png",
        model: Optional[str] = None,
    ) -> GenerationResult:
        """Upscale an image by re-rendering it at a higher size tier.

        The source image is passed as a reference together with an
        instruction to reproduce it exactly; the model outputs 2K or 4K.
        """
        use_model = model or DEFAULT_MODEL
        retired = self._retired_model_error(use_model)
        if retired is not None:
            return retired
        use_model, alias_warning = self._resolve_deprecated_alias(use_model)

        try:
            from google.genai import types

            source = Image.open(str(image_path))
            width, height = source.size
            image_size, cap_warning, error = self._upscale_target(
                width, height, upscale_factor
            )
            if error or image_size is None:
                return GenerationResult(success=False, output="", error=error)

            supported = supported_image_sizes(use_model)
            if supported is not None and image_size not in supported:
                return GenerationResult(
                    success=False,
                    output="",
                    error=(
                        f"{use_model} cannot output {image_size}. Use "
                        f"{DEFAULT_MODEL} or gemini-3-pro-image for upscaling."
                    ),
                )

            aspect_ratio = self._nearest_aspect_ratio(width, height, use_model)
            ratio_warning = self._aspect_ratio_warning(
                width, height, aspect_ratio, use_model
            )
            client = self._get_genai_client()
            config = types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                    image_size=self._sdk_image_size(image_size),
                ),
            )
            response = await client.aio.models.generate_content(
                model=use_model,
                contents=[source, UPSCALE_INSTRUCTION],
                config=config,
            )

            blocked = self._check_safety_block(response)
            if blocked is not None:
                return blocked

            images, no_image_reason = self._extract_images(response)
            if not images:
                return GenerationResult(
                    success=False,
                    output="",
                    error=no_image_reason or "Upscaling produced no output",
                )

            result_img = images[0]
            self._save_image(result_img, output_path, output_format)
            warnings: List[str] = [GENERATIVE_UPSCALE_WARNING]
            if cap_warning:
                warnings.append(cap_warning)
            if ratio_warning:
                warnings.append(ratio_warning)
            if alias_warning:
                warnings.append(alias_warning)
            return GenerationResult(
                success=True,
                output=(
                    f"Image upscaled ({upscale_factor}, {image_size}) to "
                    f"{output_path}"
                ),
                images=[result_img],
                image_paths=[str(output_path)],
                data={
                    "upscale_factor": upscale_factor,
                    "model": use_model,
                    "method": "generative",
                    "image_size": image_size,
                    "source_size": [width, height],
                    "output_size": list(result_img.size),
                    "image_path": str(output_path),
                    "warnings": warnings,
                },
            )

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
        model: Optional[str] = None,
    ) -> GenerationResult:
        """Mode-driven editing (inpaint, outpaint, background swap, ...).

        Each mode wraps the prompt in an instruction tuned for that edit.
        An optional mask is sent as a second image that guides the model.
        """
        template = EDIT_MODE_INSTRUCTIONS.get(edit_mode)
        if template is None:
            return GenerationResult(
                success=False,
                output="",
                error=f"Unknown edit mode: {edit_mode}",
            )

        use_model = model or DEFAULT_MODEL
        retired = self._retired_model_error(use_model)
        if retired is not None:
            return retired
        use_model, alias_warning = self._resolve_deprecated_alias(use_model)

        try:
            from google.genai import types

            source = Image.open(str(image_path))
            width, height = source.size

            # Match the source resolution where the model allows it.
            image_size = self._size_tier_for_edge(max(width, height))
            supported = supported_image_sizes(use_model)
            if supported is not None and image_size not in supported:
                image_size = supported[-1]

            image_config_params: Dict[str, Any] = {
                "image_size": self._sdk_image_size(image_size),
            }
            ratio_warning: Optional[str] = None
            if edit_mode != "outpaint":
                aspect_ratio = self._nearest_aspect_ratio(width, height, use_model)
                image_config_params["aspect_ratio"] = aspect_ratio
                ratio_warning = self._aspect_ratio_warning(
                    width, height, aspect_ratio, use_model
                )

            contents: List[Any] = [source]
            if mask_path:
                contents.append(Image.open(str(mask_path)))
                contents.append(MASK_INSTRUCTION)
            contents.append(template.format(prompt=prompt))

            client = self._get_genai_client()
            config = types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(**image_config_params),
            )
            response = await client.aio.models.generate_content(
                model=use_model, contents=contents, config=config
            )

            blocked = self._check_safety_block(response)
            if blocked is not None:
                return blocked

            images, no_image_reason = self._extract_images(response)
            if not images:
                return GenerationResult(
                    success=False,
                    output="",
                    error=no_image_reason or "Advanced edit produced no output",
                )

            result_img = images[0]
            self._save_image(result_img, output_path, output_format)
            warnings: List[str] = []
            if mask_path:
                warnings.append(MASK_GUIDANCE_WARNING)
            if ratio_warning:
                warnings.append(ratio_warning)
            if alias_warning:
                warnings.append(alias_warning)
            data: Dict[str, Any] = {
                "edit_mode": edit_mode,
                "model": use_model,
                "method": "generative",
                "mask_used": mask_path is not None,
                "image_size": image_size,
                "image_path": str(output_path),
            }
            if warnings:
                data["warnings"] = warnings
            return GenerationResult(
                success=True,
                output=f"Advanced edit ({edit_mode}) applied to {output_path}",
                images=[result_img],
                image_paths=[str(output_path)],
                data=data,
            )

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
