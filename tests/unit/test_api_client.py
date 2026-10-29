"""Unit tests for API client using google-genai SDK."""

import io
from unittest.mock import AsyncMock, Mock, patch

import pytest
from google.genai import errors, types
from PIL import Image

from pixelforge_mcp.utils.api_client import (
    BACKEND_GEMINI_API,
    BACKEND_VERTEX_AI,
    COMPARE_IMAGES_PROMPT,
    DEFAULT_VERTEX_LOCATION,
    EDIT_MODE_INSTRUCTIONS,
    GENERATIVE_UPSCALE_WARNING,
    MASK_GUIDANCE_WARNING,
    MASK_INSTRUCTION,
    OCR_PROMPT,
    PILLOW_FORMAT_MAP,
    RETRY_ATTEMPTS,
    RETRY_STATUS_CODES,
    SAFETY_PRESETS,
    TEXT_MODEL,
    UPSCALE_INSTRUCTION,
    GenerationResult,
    ImagenAPIClient,
)
from pixelforge_mcp.utils.validation import (
    ANALYSIS_MODELS,
    DEPRECATED_MODEL_ALIASES,
    IMAGE_MODELS,
    RETIRED_MODELS,
    supported_aspect_ratios,
)

CREDENTIAL_ENV_VARS = (
    "GOOGLE_API_KEY",
    "GOOGLE_GENERATIVE_AI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "GOOGLE_GENAI_USE_VERTEXAI",
    "GOOGLE_GENAI_USE_ENTERPRISE",
)


def _png_bytes(size=(10, 10), color="red"):
    """Encode a solid-colour PNG and return its bytes."""
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


def _image_response(img_bytes):
    """Build a generate_content response carrying one inline image."""
    part = Mock()
    part.inline_data = Mock(data=img_bytes)
    candidate = Mock()
    candidate.content.parts = [part]
    candidate.finish_reason = types.FinishReason.STOP
    response = Mock(candidates=[candidate])
    response.prompt_feedback = None
    return response


def _no_image_response(finish_reason=types.FinishReason.IMAGE_SAFETY, parts=None):
    """Build a generate_content response without an image."""
    candidate = Mock()
    candidate.content.parts = parts
    candidate.finish_reason = finish_reason
    response = Mock(candidates=[candidate])
    response.prompt_feedback = None
    return response


def _client_returning(response):
    """Return (ImagenAPIClient, mock genai client) for a canned response."""
    mock_client = Mock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=response)
    client = ImagenAPIClient(api_key="test-key")
    client._genai_client = mock_client
    return client, mock_client


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_create_successful_result(self):
        """Test creating a successful GenerationResult."""
        result = GenerationResult(
            success=True,
            output="Image generated successfully",
            error=None,
            images=[Mock(spec=Image.Image)],
            image_paths=["/test.png"],
            data={"model": "gemini-2.5-flash-image"},
        )
        assert result.success is True
        assert result.output == "Image generated successfully"
        assert result.error is None
        assert len(result.images) == 1
        assert result.image_paths == ["/test.png"]

    def test_create_failed_result(self):
        """Test creating a failed GenerationResult."""
        result = GenerationResult(success=False, output="", error="API key invalid")
        assert result.success is False
        assert result.error == "API key invalid"
        assert result.images is None
        assert result.image_paths is None


class TestImagenAPIClient:
    """Tests for ImagenAPIClient."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        client = ImagenAPIClient()
        assert client.model_name == "gemini-nano-banana-2.1"
        assert client.api_key is None
        assert client.log_images is False

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        client = ImagenAPIClient(
            model_name="gemini-3-pro-image",
            api_key="test-key",
            log_images=True,
        )
        assert client.model_name == "gemini-3-pro-image"
        assert client.api_key == "test-key"
        assert client.log_images is True

    @pytest.mark.asyncio
    async def test_generate_success(self, tmp_path):
        """Test successful image generation via direct SDK."""
        # Create a real small PNG in memory
        test_img = Image.new("RGB", (10, 10), color="red")
        buf = __import__("io").BytesIO()
        test_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        # Mock the genai client response
        mock_part = Mock()
        mock_part.inline_data = Mock(data=img_bytes)
        mock_candidate = Mock()
        mock_candidate.content.parts = [mock_part]
        mock_response = Mock(candidates=[mock_candidate])
        mock_response.prompt_feedback = None

        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client
        output_path = tmp_path / "test.png"

        result = await client.generate(
            prompt="test prompt",
            output_path=output_path,
            aspect_ratio="16:9",
            temperature=0.8,
            model="gemini-2.5-flash-image",
        )

        assert result.success is True
        assert len(result.images) == 1
        assert result.image_paths == [str(output_path)]
        assert result.data["model"] == "gemini-2.5-flash-image"
        assert result.data["aspect_ratio"] == "16:9"
        assert output_path.exists()

    @pytest.mark.asyncio
    async def test_generate_failure_no_images(self, tmp_path):
        """A response without an image reports the finish reason."""
        client, _ = _client_returning(
            _no_image_response(types.FinishReason.IMAGE_SAFETY, parts=[])
        )
        output_path = tmp_path / "test.png"

        result = await client.generate(prompt="test", output_path=output_path)

        assert result.success is False
        assert result.error == "No image returned (finish_reason=IMAGE_SAFETY)"
        assert not output_path.exists()

    @pytest.mark.asyncio
    async def test_generate_exception(self, tmp_path):
        """Test generation with exception."""
        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("API error")
        )

        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client
        output_path = tmp_path / "test.png"

        result = await client.generate(prompt="test", output_path=output_path)

        assert result.success is False
        assert "API error" in result.error

    @pytest.mark.asyncio
    async def test_edit_success(self, tmp_path):
        """Test successful image editing via direct SDK."""
        test_img = Image.new("RGB", (10, 10), color="red")
        buf = io.BytesIO()
        test_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        mock_part = Mock()
        mock_part.inline_data = Mock(data=img_bytes)
        mock_candidate = Mock()
        mock_candidate.content.parts = [mock_part]
        mock_response = Mock(candidates=[mock_candidate])
        mock_response.prompt_feedback = None

        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        input_path = tmp_path / "input.png"
        test_img.save(str(input_path))
        output_path = tmp_path / "output.png"

        result = await client.edit(
            prompt="add clouds",
            input_path=input_path,
            output_path=output_path,
            temperature=0.7,
        )

        assert result.success is True
        assert result.images is not None
        assert len(result.images) == 1
        assert result.image_paths == [str(output_path)]
        assert output_path.exists()

        # Verify generate_content was called with image + prompt
        call_kwargs = mock_client.aio.models.generate_content.call_args[1]
        assert call_kwargs["contents"][1] == "add clouds"
        assert isinstance(call_kwargs["contents"][0], Image.Image)

    @pytest.mark.asyncio
    async def test_edit_failure(self, tmp_path):
        """An edit response without an image reports the finish reason."""
        client, _ = _client_returning(
            _no_image_response(types.FinishReason.NO_IMAGE, parts=[])
        )

        input_path = tmp_path / "input.png"
        Image.new("RGB", (10, 10)).save(str(input_path))
        output_path = tmp_path / "output.png"

        result = await client.edit(
            prompt="add clouds", input_path=input_path, output_path=output_path
        )

        assert result.success is False
        assert result.error == "No image returned (finish_reason=NO_IMAGE)"
        assert not output_path.exists()

    @pytest.mark.asyncio
    async def test_analyze_success(self, tmp_path):
        """Test successful image analysis via direct SDK."""
        mock_part = Mock()
        mock_part.text = "A beautiful sunset over mountains"
        mock_part.inline_data = None
        mock_candidate = Mock()
        mock_candidate.content.parts = [mock_part]
        mock_response = Mock(candidates=[mock_candidate])
        mock_response.prompt_feedback = None

        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        image_path = tmp_path / "test.png"
        Image.new("RGB", (10, 10)).save(str(image_path))

        result = await client.analyze(image_path)

        assert result.success is True
        assert result.output == "A beautiful sunset over mountains"
        assert result.data["analysis"] == "A beautiful sunset over mountains"
        assert result.data["image_path"] == str(image_path)

    @pytest.mark.asyncio
    async def test_analyze_with_custom_prompt(self, tmp_path):
        """Test image analysis with a custom prompt."""
        mock_part = Mock()
        mock_part.text = "Text found: Hello World"
        mock_part.inline_data = None
        mock_candidate = Mock()
        mock_candidate.content.parts = [mock_part]
        mock_response = Mock(candidates=[mock_candidate])
        mock_response.prompt_feedback = None

        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        image_path = tmp_path / "test.png"
        Image.new("RGB", (10, 10)).save(str(image_path))

        custom_prompt = "Extract all visible text from this image"
        result = await client.analyze(image_path, prompt=custom_prompt)

        assert result.success is True
        assert result.output == "Text found: Hello World"

        # Verify custom prompt was passed in contents
        call_kwargs = mock_client.aio.models.generate_content.call_args[1]
        contents = call_kwargs["contents"]
        assert contents[-1] == custom_prompt

    @pytest.mark.asyncio
    async def test_analyze_failure(self, tmp_path):
        """Test failed image analysis with empty response."""
        mock_candidate = Mock()
        mock_candidate.content.parts = []
        mock_response = Mock(candidates=[mock_candidate])
        mock_response.prompt_feedback = None

        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        image_path = tmp_path / "test.png"
        Image.new("RGB", (10, 10)).save(str(image_path))

        result = await client.analyze(image_path)

        assert result.success is False
        assert result.error == "No analysis generated"

    @pytest.mark.asyncio
    async def test_list_models_success(self):
        """Test successful model listing with metadata."""
        client = ImagenAPIClient()

        result = await client.list_models()

        assert result.success is True
        assert "models" in result.data
        assert "recommendation" in result.data

        models = result.data["models"]
        assert isinstance(models, list)
        assert len(models) == 5

        model_names = [m["name"] for m in models]
        assert model_names == [
            "gemini-nano-banana-2.1",
            "gemini-3.1-flash-lite-image",
            "gemini-3.1-flash-image",
            "gemini-3-pro-image",
            "gemini-2.5-flash-image",
        ]
        assert model_names == IMAGE_MODELS
        assert not any(name.startswith("imagen-") for name in model_names)
        assert not any(name.endswith("-preview") for name in model_names)

        for model in models:
            assert "name" in model
            assert "nickname" in model
            assert "speed" in model
            assert "quality" in model
            assert "description" in model
            assert "best_for" in model
            assert "capabilities" in model
            assert "temperature" in model
            assert "pricing" in model
            assert "note" not in model

    @pytest.mark.asyncio
    async def test_list_models_nano_banana_2_1_metadata(self):
        """Test Nano Banana 2.1 model metadata is correct."""
        client = ImagenAPIClient()
        result = await client.list_models()

        models = result.data["models"]
        nb21 = next(m for m in models if m["name"] == "gemini-nano-banana-2.1")

        assert nb21["nickname"] == "Nano Banana 2.1"
        assert nb21["default"] is True
        assert nb21["quality"] == "excellent"
        assert nb21["capabilities"]["panoramic"] is True
        assert nb21["capabilities"]["reference_images"] == 14
        assert nb21["capabilities"]["resolution"] == "1K/2K/4K"
        assert nb21["capabilities"]["person_generation"] is False
        assert nb21["capabilities"]["grounding"]
        assert nb21["capabilities"]["thinking"]
        assert nb21["temperature"] == {"min": 0.0, "max": 2.0, "default": 1.0}
        assert nb21["pricing"]["1K"] == 0.0336
        assert nb21["pricing"]["2K"] == 0.0504
        assert nb21["pricing"]["4K"] == 0.113
        assert "deprecated" not in nb21

    @pytest.mark.asyncio
    async def test_list_models_only_default_is_nano_banana_2_1(self):
        """Test exactly one model is flagged default."""
        client = ImagenAPIClient()
        result = await client.list_models()

        defaults = [m["name"] for m in result.data["models"] if m["default"]]
        assert defaults == ["gemini-nano-banana-2.1"]

    @pytest.mark.asyncio
    async def test_list_models_flash_lite_is_1k_without_panoramic(self):
        """Test flash-lite reports 1K only, no panoramic, no grounding."""
        client = ImagenAPIClient()
        result = await client.list_models()

        lite = next(
            m
            for m in result.data["models"]
            if m["name"] == "gemini-3.1-flash-lite-image"
        )

        assert lite["nickname"] == "Nano Banana 2 Lite"
        assert lite["capabilities"]["panoramic"] is False
        assert lite["capabilities"]["grounding"] is False
        assert lite["capabilities"]["resolution"] == "1K"
        # Google gives no reference-image limit for this model.
        assert lite["capabilities"]["reference_images"] is None
        assert lite["pricing"]["1K"] == 0.0336
        assert lite["pricing"]["2K"] is None
        assert lite["pricing"]["4K"] is None

    @pytest.mark.asyncio
    async def test_list_models_flash_image_pricing_and_sizes(self):
        """Test gemini-3.1-flash-image reports the 0.5K tier and its prices."""
        client = ImagenAPIClient()
        result = await client.list_models()

        flash = next(
            m for m in result.data["models"] if m["name"] == "gemini-3.1-flash-image"
        )

        assert flash["capabilities"]["panoramic"] is True
        assert flash["capabilities"]["resolution"] == "0.5K/1K/2K/4K"
        assert flash["pricing"] == {
            "0.5K": 0.045,
            "1K": 0.067,
            "2K": 0.101,
            "4K": 0.151,
        }

    @pytest.mark.asyncio
    async def test_list_models_pro_metadata(self):
        """Test Nano Banana Pro metadata."""
        client = ImagenAPIClient()
        result = await client.list_models()

        pro = next(
            m for m in result.data["models"] if m["name"] == "gemini-3-pro-image"
        )

        assert pro["nickname"] == "Nano Banana Pro"
        assert pro["capabilities"]["reference_images"] == 6
        assert pro["capabilities"]["panoramic"] is False
        assert pro["pricing"]["1K"] == 0.134
        assert pro["pricing"]["2K"] == 0.134
        assert pro["pricing"]["4K"] == 0.24

    @pytest.mark.asyncio
    async def test_list_models_marks_legacy_model_deprecated(self):
        """Test gemini-2.5-flash-image is deprecated with a replacement."""
        client = ImagenAPIClient()
        result = await client.list_models()

        legacy = next(
            m for m in result.data["models"] if m["name"] == "gemini-2.5-flash-image"
        )

        assert legacy["deprecated"] is True
        assert legacy["replacement"] == "gemini-3.1-flash-lite-image"
        assert legacy["default"] is False
        assert result.data["models"][-1]["name"] == "gemini-2.5-flash-image"

        others = [m for m in result.data["models"] if m is not legacy]
        assert all(not m.get("deprecated") for m in others)

    @pytest.mark.asyncio
    async def test_list_models_includes_analysis_models(self):
        """Test list_models returns the text analysis models."""
        client = ImagenAPIClient()
        result = await client.list_models()

        assert "analysis_models" in result.data
        analysis = result.data["analysis_models"]
        assert [m["name"] for m in analysis] == ANALYSIS_MODELS
        assert [m["name"] for m in analysis] == [
            "gemini-3.5-flash-lite",
            "gemini-3.8-flash",
        ]

        defaults = [m["name"] for m in analysis if m["default"]]
        assert defaults == ["gemini-3.5-flash-lite"]

        for model in analysis:
            assert "nickname" in model
            assert "description" in model
            assert "best_for" in model
            assert set(model["pricing_per_1m_tokens"]) == {"input", "output"}

        lite = analysis[0]
        assert lite["pricing_per_1m_tokens"] == {"input": 0.30, "output": 2.50}
        strongest = analysis[1]
        assert strongest["pricing_per_1m_tokens"] == {"input": 0.75, "output": 3.75}

    @pytest.mark.asyncio
    async def test_list_models_recommendation_names_presets(self):
        """Test the recommendation points at the stable models."""
        client = ImagenAPIClient()
        result = await client.list_models()

        recommendation = result.data["recommendation"]
        assert "gemini-3.1-flash-lite-image" in recommendation
        assert "gemini-nano-banana-2.1" in recommendation
        assert "gemini-3-pro-image" in recommendation
        assert "gemini-3.5-flash-lite" in recommendation
        assert "gemini-3.8-flash" in recommendation
        assert "imagen" not in recommendation.lower()

    @pytest.mark.asyncio
    async def test_list_models_person_generation_capability(self):
        """Test no model reports API-level person_generation control."""
        client = ImagenAPIClient()
        result = await client.list_models()

        for model in result.data["models"]:
            assert model["capabilities"]["person_generation"] is False

    @pytest.mark.asyncio
    async def test_generate_model_override(self, tmp_path):
        """Test model override is passed to SDK call."""
        test_img = Image.new("RGB", (10, 10), color="red")
        buf = __import__("io").BytesIO()
        test_img.save(buf, format="PNG")

        mock_part = Mock()
        mock_part.inline_data = Mock(data=buf.getvalue())
        mock_candidate = Mock()
        mock_candidate.content.parts = [mock_part]
        mock_response = Mock(candidates=[mock_candidate])
        mock_response.prompt_feedback = None

        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        await client.generate(
            prompt="test",
            output_path=tmp_path / "test.png",
            model="gemini-3-pro-image",
        )

        # Verify the model was passed to the SDK call
        call_kwargs = mock_client.aio.models.generate_content.call_args[1]
        assert call_kwargs["model"] == "gemini-3-pro-image"


class TestSafetySettings:
    """Tests for safety settings conversion."""

    def test_build_safety_strict(self):
        """Test strict safety preset builds correct settings."""
        settings = ImagenAPIClient._build_safety_settings("preset:strict")
        assert settings is not None
        assert len(settings) == 4

    def test_build_safety_relaxed(self):
        """Test relaxed safety preset builds correct settings."""
        settings = ImagenAPIClient._build_safety_settings("preset:relaxed")
        assert settings is not None
        assert len(settings) == 4

    def test_build_safety_none_input(self):
        """Test None input returns None."""
        assert ImagenAPIClient._build_safety_settings(None) is None

    def test_build_safety_non_preset(self):
        """Test non-preset string returns None."""
        assert ImagenAPIClient._build_safety_settings("custom:value") is None

    def test_build_safety_invalid_preset(self):
        """Test invalid preset name returns None."""
        assert ImagenAPIClient._build_safety_settings("preset:invalid") is None


class TestSaveImage:
    """Tests for _save_image static method."""

    def test_save_png(self, tmp_path):
        """Test saving image as PNG."""
        img = Image.new("RGB", (100, 100), color="red")
        output_path = tmp_path / "test.png"

        ImagenAPIClient._save_image(img, output_path, "png")

        assert output_path.exists()
        saved = Image.open(str(output_path))
        assert saved.format == "PNG"

    def test_save_jpeg(self, tmp_path):
        """Test saving image as JPEG."""
        img = Image.new("RGB", (100, 100), color="red")
        output_path = tmp_path / "test.jpg"

        ImagenAPIClient._save_image(img, output_path, "jpeg")

        assert output_path.exists()
        saved = Image.open(str(output_path))
        assert saved.format == "JPEG"

    def test_save_webp(self, tmp_path):
        """Test saving image as WebP."""
        img = Image.new("RGB", (100, 100), color="red")
        output_path = tmp_path / "test.webp"

        ImagenAPIClient._save_image(img, output_path, "webp")

        assert output_path.exists()
        saved = Image.open(str(output_path))
        assert saved.format == "WEBP"

    def test_save_jpeg_converts_rgba_with_white_bg(self, tmp_path):
        """Test JPEG saving composites RGBA onto white background."""
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 0))
        output_path = tmp_path / "test.jpg"

        ImagenAPIClient._save_image(img, output_path, "jpeg")

        assert output_path.exists()
        saved = Image.open(str(output_path))
        assert saved.mode == "RGB"
        # Fully transparent red should become white (not black)
        pixel = saved.getpixel((50, 50))
        assert pixel == (255, 255, 255)

    def test_save_creates_parent_dirs(self, tmp_path):
        """Test saving creates parent directories."""
        output_path = tmp_path / "subdir" / "deep" / "test.png"
        img = Image.new("RGB", (100, 100), color="red")

        ImagenAPIClient._save_image(img, output_path, "png")

        assert output_path.exists()


class TestEditWithModel:
    """Tests for edit with model override."""

    @pytest.mark.asyncio
    async def test_edit_with_model_override(self, tmp_path):
        """Test edit passes model override to generate_content."""
        test_img = Image.new("RGB", (10, 10), color="red")
        buf = io.BytesIO()
        test_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        mock_part = Mock()
        mock_part.inline_data = Mock(data=img_bytes)
        mock_candidate = Mock()
        mock_candidate.content.parts = [mock_part]
        mock_response = Mock(candidates=[mock_candidate])
        mock_response.prompt_feedback = None

        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        input_path = tmp_path / "input.png"
        test_img.save(str(input_path))
        output_path = tmp_path / "output.png"

        result = await client.edit(
            prompt="add clouds",
            input_path=input_path,
            output_path=output_path,
            model="gemini-3-pro-image",
        )

        assert result.success is True
        # Verify the model was passed to the SDK call
        call_kwargs = mock_client.aio.models.generate_content.call_args[1]
        assert call_kwargs["model"] == "gemini-3-pro-image"

    @pytest.mark.asyncio
    async def test_edit_without_model_uses_default(self, tmp_path):
        """Test edit without model uses the client default."""
        test_img = Image.new("RGB", (10, 10), color="red")
        buf = io.BytesIO()
        test_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        mock_part = Mock()
        mock_part.inline_data = Mock(data=img_bytes)
        mock_candidate = Mock()
        mock_candidate.content.parts = [mock_part]
        mock_response = Mock(candidates=[mock_candidate])
        mock_response.prompt_feedback = None

        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = ImagenAPIClient(
            model_name="gemini-2.5-flash-image", api_key="test-key"
        )
        client._genai_client = mock_client

        input_path = tmp_path / "input.png"
        test_img.save(str(input_path))
        output_path = tmp_path / "output.png"

        result = await client.edit(
            prompt="add clouds",
            input_path=input_path,
            output_path=output_path,
        )

        assert result.success is True
        assert result.data["model"] == "gemini-2.5-flash-image"


class TestConstants:
    """Tests for api_client constants."""

    def test_pillow_format_map(self):
        """Test Pillow format map has all formats."""
        assert PILLOW_FORMAT_MAP["png"] == "PNG"
        assert PILLOW_FORMAT_MAP["jpeg"] == "JPEG"
        assert PILLOW_FORMAT_MAP["webp"] == "WEBP"

    def test_text_model_constant(self):
        """Test text model is defined."""
        from pixelforge_mcp.utils.api_client import TEXT_MODEL

        assert TEXT_MODEL == "gemini-3.5-flash-lite"

    def test_safety_presets(self):
        """Test safety presets are defined."""
        assert "strict" in SAFETY_PRESETS
        assert "relaxed" in SAFETY_PRESETS
        assert "none" in SAFETY_PRESETS


class TestJsonParsing:
    """Tests for _parse_json_from_text helper."""

    def test_parse_plain_json(self):
        """Test parsing plain JSON."""
        result = ImagenAPIClient._parse_json_from_text('{"text": "hello"}')
        assert result == {"text": "hello"}

    def test_parse_json_with_code_fence(self):
        """Test parsing JSON wrapped in markdown code fences."""
        text = '```json\n{"text": "hello"}\n```'
        result = ImagenAPIClient._parse_json_from_text(text)
        assert result == {"text": "hello"}

    def test_parse_json_array(self):
        """Test parsing JSON array."""
        text = '[{"label": "cat", "box_2d": [10, 20, 30, 40]}]'
        result = ImagenAPIClient._parse_json_from_text(text)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_parse_invalid_json(self):
        """Test invalid JSON returns None."""
        result = ImagenAPIClient._parse_json_from_text("not json at all")
        assert result is None

    def test_parse_empty_string(self):
        """Test empty string returns None."""
        result = ImagenAPIClient._parse_json_from_text("")
        assert result is None


class TestGenerateAlwaysUsesSDK:
    """Tests verifying generate always routes through direct SDK."""

    @pytest.mark.asyncio
    async def test_generate_uses_sdk_even_without_extended_params(self, tmp_path):
        """Test basic generate call goes through SDK (no dual path)."""
        test_img = Image.new("RGB", (10, 10), color="blue")
        buf = __import__("io").BytesIO()
        test_img.save(buf, format="PNG")

        mock_part = Mock()
        mock_part.inline_data = Mock(data=buf.getvalue())
        mock_candidate = Mock()
        mock_candidate.content.parts = [mock_part]
        mock_response = Mock(candidates=[mock_candidate])
        mock_response.prompt_feedback = None

        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        result = await client.generate(
            prompt="simple test",
            output_path=tmp_path / "out.png",
        )

        assert result.success is True
        # Verify SDK was called (not gemini-imagen)
        mock_client.aio.models.generate_content.assert_called_once()


class TestExtractText:
    """Tests for extract_text method."""

    @pytest.mark.asyncio
    async def test_extract_text_success_json(self, tmp_path):
        """Test successful text extraction with valid JSON response."""
        client = ImagenAPIClient(api_key="test-key")
        image_path = tmp_path / "test.png"
        image_path.touch()

        json_response = (
            '{"text": "Hello World", '
            '"blocks": [{"text": "Hello World", "confidence": "high"}]}'
        )
        client._call_text_model_with_image = AsyncMock(return_value=json_response)

        result = await client.extract_text(image_path)

        assert result.success is True
        assert result.output == "Hello World"
        assert result.data["extracted_text"] == "Hello World"
        assert len(result.data["blocks"]) == 1
        assert result.data["blocks"][0]["confidence"] == "high"
        assert result.data["raw_response"] == json_response
        assert result.data["image_path"] == str(image_path)
        client._call_text_model_with_image.assert_called_once_with(
            image_path, OCR_PROMPT
        )

    @pytest.mark.asyncio
    async def test_extract_text_fallback_raw_text(self, tmp_path):
        """Test text extraction falls back to raw text when JSON parsing fails."""
        client = ImagenAPIClient(api_key="test-key")
        image_path = tmp_path / "test.png"
        image_path.touch()

        raw_response = "Some plain text without JSON"
        client._call_text_model_with_image = AsyncMock(return_value=raw_response)

        result = await client.extract_text(image_path)

        assert result.success is True
        assert result.output == raw_response
        assert result.data["extracted_text"] == raw_response
        assert result.data["blocks"] == []
        assert result.data["raw_response"] == raw_response

    @pytest.mark.asyncio
    async def test_extract_text_exception(self, tmp_path):
        """Test text extraction handles exceptions."""
        client = ImagenAPIClient(api_key="test-key")
        image_path = tmp_path / "test.png"
        image_path.touch()

        client._call_text_model_with_image = AsyncMock(
            side_effect=Exception("API error")
        )

        result = await client.extract_text(image_path)

        assert result.success is False
        assert "API error" in result.error


class TestDetectObjects:
    """Tests for detect_objects method."""

    @pytest.mark.asyncio
    async def test_detect_objects_list_result(self, tmp_path):
        """Test detection with JSON array response."""
        client = ImagenAPIClient(api_key="test-key")
        image_path = tmp_path / "test.png"
        image_path.touch()

        json_response = (
            '[{"label": "cat", "box_2d": [100, 200, 300, 400], '
            '"confidence": "high"}, '
            '{"label": "dog", "box_2d": [500, 600, 700, 800], '
            '"confidence": "medium"}]'
        )
        client._call_text_model_with_image = AsyncMock(return_value=json_response)

        result = await client.detect_objects(image_path)

        assert result.success is True
        assert result.output == "Detected 2 object(s)"
        assert result.data["count"] == 2
        assert len(result.data["detections"]) == 2
        assert result.data["detections"][0]["label"] == "cat"
        assert result.data["detections"][1]["label"] == "dog"

    @pytest.mark.asyncio
    async def test_detect_objects_dict_result(self, tmp_path):
        """Test detection wraps a dict response in a list."""
        client = ImagenAPIClient(api_key="test-key")
        image_path = tmp_path / "test.png"
        image_path.touch()

        json_response = (
            '{"label": "cat", "box_2d": [100, 200, 300, 400], ' '"confidence": "high"}'
        )
        client._call_text_model_with_image = AsyncMock(return_value=json_response)

        result = await client.detect_objects(image_path)

        assert result.success is True
        assert result.output == "Detected 1 object(s)"
        assert result.data["count"] == 1
        assert result.data["detections"][0]["label"] == "cat"

    @pytest.mark.asyncio
    async def test_detect_objects_with_target(self, tmp_path):
        """Test detection with specific objects parameter."""
        client = ImagenAPIClient(api_key="test-key")
        image_path = tmp_path / "test.png"
        image_path.touch()

        json_response = (
            '[{"label": "cat", "box_2d": [100, 200, 300, 400], '
            '"confidence": "high"}]'
        )
        client._call_text_model_with_image = AsyncMock(return_value=json_response)

        result = await client.detect_objects(image_path, objects="cats and dogs")

        assert result.success is True
        # Verify the prompt used the specific target
        call_args = client._call_text_model_with_image.call_args
        prompt_used = call_args[0][1]
        assert "cats and dogs" in prompt_used

    @pytest.mark.asyncio
    async def test_detect_objects_empty(self, tmp_path):
        """Test detection with non-JSON response returns empty detections."""
        client = ImagenAPIClient(api_key="test-key")
        image_path = tmp_path / "test.png"
        image_path.touch()

        client._call_text_model_with_image = AsyncMock(
            return_value="No objects found in this image"
        )

        result = await client.detect_objects(image_path)

        assert result.success is True
        assert result.data["detections"] == []
        assert result.data["count"] == 0
        assert result.data["raw_response"] == "No objects found in this image"

    @pytest.mark.asyncio
    async def test_detect_objects_exception(self, tmp_path):
        """Test detection handles exceptions."""
        client = ImagenAPIClient(api_key="test-key")
        image_path = tmp_path / "test.png"
        image_path.touch()

        client._call_text_model_with_image = AsyncMock(
            side_effect=Exception("Connection error")
        )

        result = await client.detect_objects(image_path)

        assert result.success is False
        assert "Connection error" in result.error


class TestCompareImages:
    """Tests for compare_images method."""

    @pytest.mark.asyncio
    async def test_compare_images_default_prompt(self, tmp_path):
        """Test compare_images uses default prompt when none provided."""
        client = ImagenAPIClient(api_key="test-key")

        mock_result = GenerationResult(
            success=True,
            output="The images are very similar",
            data={"analysis": "The images are very similar"},
        )
        client._analyze_with_images = AsyncMock(return_value=mock_result)

        image_paths = [tmp_path / "img1.png", tmp_path / "img2.png"]

        result = await client.compare_images(image_paths)

        assert result.success is True
        assert result.output == "The images are very similar"
        client._analyze_with_images.assert_called_once_with(
            image_paths, COMPARE_IMAGES_PROMPT
        )

    @pytest.mark.asyncio
    async def test_compare_images_custom_prompt(self, tmp_path):
        """Test compare_images uses custom prompt when provided."""
        client = ImagenAPIClient(api_key="test-key")

        mock_result = GenerationResult(
            success=True,
            output="Custom comparison result",
            data={"analysis": "Custom comparison result"},
        )
        client._analyze_with_images = AsyncMock(return_value=mock_result)

        image_paths = [tmp_path / "img1.png", tmp_path / "img2.png"]
        custom_prompt = "Which image has more blue?"

        result = await client.compare_images(image_paths, prompt=custom_prompt)

        assert result.success is True
        client._analyze_with_images.assert_called_once_with(image_paths, custom_prompt)


class TestOptimizePrompt:
    """Tests for optimize_prompt method."""

    @pytest.mark.asyncio
    async def test_optimize_with_style(self):
        """Test prompt optimization with a style parameter."""
        client = ImagenAPIClient(api_key="test-key")
        client._call_text_model = AsyncMock(
            return_value='"A photorealistic sunset over mountains"'
        )

        result = await client.optimize_prompt(
            prompt="sunset over mountains", style="photorealistic"
        )

        assert result.success is True
        assert result.output == "A photorealistic sunset over mountains"
        assert result.data["original_prompt"] == "sunset over mountains"
        assert result.data["enhanced_prompt"] == (
            "A photorealistic sunset over mountains"
        )
        assert result.data["style"] == "photorealistic"

        # Verify style instruction was included in the meta prompt
        call_args = client._call_text_model.call_args[0][0]
        assert "photorealistic" in call_args

    @pytest.mark.asyncio
    async def test_optimize_without_style(self):
        """Test prompt optimization without a style parameter."""
        client = ImagenAPIClient(api_key="test-key")
        client._call_text_model = AsyncMock(
            return_value="Enhanced: a vivid sunset with dramatic lighting"
        )

        result = await client.optimize_prompt(prompt="sunset")

        assert result.success is True
        assert result.output == "Enhanced: a vivid sunset with dramatic lighting"
        assert result.data["original_prompt"] == "sunset"
        assert result.data["style"] is None

    @pytest.mark.asyncio
    async def test_optimize_empty_response(self):
        """Test prompt optimization fails on empty response."""
        client = ImagenAPIClient(api_key="test-key")
        client._call_text_model = AsyncMock(return_value="")

        result = await client.optimize_prompt(prompt="sunset")

        assert result.success is False
        assert result.error == "Prompt optimization failed"

    @pytest.mark.asyncio
    async def test_optimize_exception(self):
        """Test prompt optimization handles exceptions."""
        client = ImagenAPIClient(api_key="test-key")
        client._call_text_model = AsyncMock(
            side_effect=Exception("Rate limit exceeded")
        )

        result = await client.optimize_prompt(prompt="sunset")

        assert result.success is False
        assert "Rate limit exceeded" in result.error


class TestRemoveBackground:
    """Tests for remove_background method."""

    @pytest.mark.asyncio
    async def test_remove_bg_success(self, tmp_path):
        """Test successful background removal via direct SDK."""
        test_img = Image.new("RGB", (10, 10), color="red")
        buf = io.BytesIO()
        test_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        mock_part = Mock()
        mock_part.inline_data = Mock(data=img_bytes)
        mock_candidate = Mock()
        mock_candidate.content.parts = [mock_part]
        mock_response = Mock(candidates=[mock_candidate])
        mock_response.prompt_feedback = None

        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        image_path = tmp_path / "input.png"
        test_img.save(str(image_path))
        output_path = tmp_path / "output.png"

        result = await client.remove_background(image_path, output_path)

        assert result.success is True
        assert "Background removed" in result.output
        assert result.images is not None
        assert result.image_paths == [str(output_path)]
        assert result.data["image_path"] == str(output_path)
        assert output_path.exists()

    @pytest.mark.asyncio
    async def test_remove_bg_format_conversion(self, tmp_path):
        """Test background removal with format conversion."""
        test_img = Image.new("RGB", (10, 10), color="red")
        buf = io.BytesIO()
        test_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        mock_part = Mock()
        mock_part.inline_data = Mock(data=img_bytes)
        mock_candidate = Mock()
        mock_candidate.content.parts = [mock_part]
        mock_response = Mock(candidates=[mock_candidate])
        mock_response.prompt_feedback = None

        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        image_path = tmp_path / "input.png"
        test_img.save(str(image_path))
        output_path = tmp_path / "output.jpg"

        result = await client.remove_background(
            image_path, output_path, output_format="jpeg"
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_remove_bg_failure(self, tmp_path):
        """A response without an image passes the edit error through."""
        client, _ = _client_returning(
            _no_image_response(types.FinishReason.IMAGE_SAFETY, parts=[])
        )

        image_path = tmp_path / "input.png"
        Image.new("RGB", (10, 10)).save(str(image_path))
        output_path = tmp_path / "output.png"

        result = await client.remove_background(image_path, output_path)

        assert result.success is False
        assert result.error == "No image returned (finish_reason=IMAGE_SAFETY)"
        assert result.data is None

    @pytest.mark.asyncio
    async def test_remove_bg_exception(self, tmp_path):
        """Test background removal handles exceptions."""
        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("Service unavailable")
        )

        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        image_path = tmp_path / "input.png"
        Image.new("RGB", (10, 10)).save(str(image_path))
        output_path = tmp_path / "output.png"

        result = await client.remove_background(image_path, output_path)

        assert result.success is False
        assert "Service unavailable" in result.error


class TestCallTextModel:
    """Tests for _call_text_model private method."""

    @pytest.mark.asyncio
    async def test_call_text_model_success(self):
        """Test successful text model call."""
        mock_part = Mock()
        mock_part.text = "Generated response text"
        mock_candidate = Mock()
        mock_candidate.content.parts = [mock_part]
        mock_response = Mock(candidates=[mock_candidate])
        mock_response.prompt_feedback = None

        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        result = await client._call_text_model("test prompt")

        assert result == "Generated response text"
        call_kwargs = mock_client.aio.models.generate_content.call_args[1]
        assert call_kwargs["model"] == TEXT_MODEL
        assert call_kwargs["contents"] == "test prompt"

    @pytest.mark.asyncio
    async def test_call_text_model_empty(self):
        """Test text model returns empty string when no candidates."""
        mock_response = Mock(candidates=[])
        mock_response.prompt_feedback = None

        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        result = await client._call_text_model("test prompt")

        assert result == ""

    @pytest.mark.asyncio
    async def test_call_text_model_empty_parts(self):
        """Test text model returns empty string when parts list is empty."""
        mock_candidate = Mock()
        mock_candidate.content.parts = []
        mock_response = Mock(candidates=[mock_candidate])
        mock_response.prompt_feedback = None

        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        result = await client._call_text_model("test prompt")

        assert result == ""


class TestCallTextModelWithImage:
    """Tests for _call_text_model_with_image private method."""

    @pytest.mark.asyncio
    async def test_with_image_success(self, tmp_path):
        """Test successful text model call with image."""
        # Create a real small image for PIL.Image.open
        test_img = Image.new("RGB", (10, 10), color="blue")
        image_path = tmp_path / "test.png"
        test_img.save(str(image_path))

        mock_part = Mock()
        mock_part.text = "Description of the image"
        mock_candidate = Mock()
        mock_candidate.content.parts = [mock_part]
        mock_response = Mock(candidates=[mock_candidate])
        mock_response.prompt_feedback = None

        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        result = await client._call_text_model_with_image(
            image_path, "Describe this image"
        )

        assert result == "Description of the image"
        call_kwargs = mock_client.aio.models.generate_content.call_args[1]
        assert call_kwargs["model"] == TEXT_MODEL
        # Contents should be [image, prompt]
        contents = call_kwargs["contents"]
        assert len(contents) == 2
        assert isinstance(contents[0], Image.Image)
        assert contents[1] == "Describe this image"

    @pytest.mark.asyncio
    async def test_with_image_empty(self, tmp_path):
        """Test text model with image returns empty on no candidates."""
        test_img = Image.new("RGB", (10, 10), color="green")
        image_path = tmp_path / "test.png"
        test_img.save(str(image_path))

        mock_response = Mock(candidates=[])
        mock_response.prompt_feedback = None

        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        result = await client._call_text_model_with_image(image_path, "Describe this")

        assert result == ""

    @pytest.mark.asyncio
    async def test_with_image_none_text(self, tmp_path):
        """Test text model with image returns empty when text is None."""
        test_img = Image.new("RGB", (10, 10), color="red")
        image_path = tmp_path / "test.png"
        test_img.save(str(image_path))

        mock_part = Mock()
        mock_part.text = None
        mock_candidate = Mock()
        mock_candidate.content.parts = [mock_part]
        mock_response = Mock(candidates=[mock_candidate])
        mock_response.prompt_feedback = None

        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        result = await client._call_text_model_with_image(image_path, "Describe this")

        assert result == ""


class TestPersonGenerationWarnOnly:
    """Tests for person_generation warn-only behavior on Gemini models."""

    @pytest.mark.asyncio
    async def test_gemini_preserves_model_with_person_generation(self, tmp_path):
        """person_generation on Gemini stays on Gemini (no auto-routing)."""
        test_img = Image.new("RGB", (10, 10), color="red")
        buf = io.BytesIO()
        test_img.save(buf, format="PNG")

        mock_part = Mock()
        mock_part.inline_data = Mock(data=buf.getvalue())
        mock_candidate = Mock()
        mock_candidate.content.parts = [mock_part]
        mock_response = Mock(candidates=[mock_candidate])
        mock_response.prompt_feedback = None

        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client
        output_path = tmp_path / "out.png"

        result = await client.generate(
            prompt="test image",
            output_path=output_path,
            person_generation="block",
        )

        assert result.success is True
        # Stayed on Gemini — generate_content called, NOT generate_images
        mock_client.aio.models.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_gemini_warns_on_person_generation(self, tmp_path):
        """person_generation on Gemini produces a warning."""
        test_img = Image.new("RGB", (10, 10), color="red")
        buf = io.BytesIO()
        test_img.save(buf, format="PNG")

        mock_part = Mock()
        mock_part.inline_data = Mock(data=buf.getvalue())
        mock_candidate = Mock()
        mock_candidate.content.parts = [mock_part]
        mock_response = Mock(candidates=[mock_candidate])
        mock_response.prompt_feedback = None

        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client
        output_path = tmp_path / "out.png"

        result = await client.generate(
            prompt="test image",
            output_path=output_path,
            person_generation="block",
        )

        assert result.success is True
        assert result.data is not None
        assert "warnings" in result.data
        assert len(result.data["warnings"]) == 1
        warning = result.data["warnings"][0]
        assert "person_generation='block'" in warning
        assert "not enforced" in warning
        assert "Imagen" not in warning
        assert "imagen" not in warning
        assert result.data["person_generation"] is None


class TestAdvancedEdit:
    """Tests for mode-driven editing via generate_content."""

    @staticmethod
    def _write_png(path, size=(10, 10), color="red"):
        """Write a small real PNG to ``path``."""
        Image.new("RGB", size, color=color).save(str(path))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", sorted(EDIT_MODE_INSTRUCTIONS))
    async def test_mode_template_wraps_prompt(self, tmp_path, mode):
        """Each mode sends the source image plus its instruction template."""
        image_path = tmp_path / "input.png"
        self._write_png(image_path)
        output_path = tmp_path / "edited.png"
        client, mock_client = _client_returning(_image_response(_png_bytes()))

        result = await client.advanced_edit(
            image_path=image_path,
            prompt="a red balloon",
            edit_mode=mode,
            output_path=output_path,
        )

        assert result.success is True
        assert output_path.exists()
        mock_client.aio.models.generate_content.assert_awaited_once()
        kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        assert kwargs["model"] == "gemini-nano-banana-2.1"
        source, instruction = kwargs["contents"]
        assert isinstance(source, Image.Image)
        assert instruction == EDIT_MODE_INSTRUCTIONS[mode].format(
            prompt="a red balloon"
        )
        assert "a red balloon" in instruction

        data = result.data
        assert data["edit_mode"] == mode
        assert data["model"] == "gemini-nano-banana-2.1"
        assert data["method"] == "generative"
        assert data["mask_used"] is False
        assert data["image_size"] == "1K"
        assert data["image_path"] == str(output_path)
        assert "warnings" not in data

    @pytest.mark.asyncio
    async def test_mask_is_second_image_followed_by_instruction(self, tmp_path):
        """A mask goes after the source image, then the mask instruction."""
        image_path = tmp_path / "input.png"
        mask_path = tmp_path / "mask.png"
        self._write_png(image_path)
        self._write_png(mask_path, color="white")
        client, mock_client = _client_returning(_image_response(_png_bytes()))

        result = await client.advanced_edit(
            image_path=image_path,
            prompt="the object",
            edit_mode="inpaint_removal",
            output_path=tmp_path / "edited.png",
            mask_path=mask_path,
        )

        assert result.success is True
        contents = mock_client.aio.models.generate_content.call_args.kwargs["contents"]
        assert len(contents) == 4
        assert isinstance(contents[0], Image.Image)
        assert isinstance(contents[1], Image.Image)
        assert contents[2] == MASK_INSTRUCTION
        assert contents[3] == EDIT_MODE_INSTRUCTIONS["inpaint_removal"].format(
            prompt="the object"
        )
        assert result.data["mask_used"] is True
        assert result.data["warnings"] == [MASK_GUIDANCE_WARNING]

    @pytest.mark.asyncio
    async def test_outpaint_leaves_aspect_ratio_free(self, tmp_path):
        """Outpainting must not pin the output to the source aspect ratio."""
        image_path = tmp_path / "input.png"
        self._write_png(image_path, size=(1600, 900))
        client, mock_client = _client_returning(_image_response(_png_bytes()))

        result = await client.advanced_edit(
            image_path=image_path,
            prompt="more sky",
            edit_mode="outpaint",
            output_path=tmp_path / "edited.png",
        )

        assert result.success is True
        config = mock_client.aio.models.generate_content.call_args.kwargs["config"]
        assert config.image_config.aspect_ratio is None
        assert config.image_config.image_size == "2K"

    @pytest.mark.asyncio
    async def test_other_modes_pin_aspect_ratio_to_source(self, tmp_path):
        """Non-outpaint modes keep the source's nearest supported ratio."""
        image_path = tmp_path / "input.png"
        self._write_png(image_path, size=(1600, 900))
        client, mock_client = _client_returning(_image_response(_png_bytes()))

        result = await client.advanced_edit(
            image_path=image_path,
            prompt="a beach",
            edit_mode="background_swap",
            output_path=tmp_path / "edited.png",
        )

        assert result.success is True
        config = mock_client.aio.models.generate_content.call_args.kwargs["config"]
        assert config.image_config.aspect_ratio == "16:9"
        assert config.image_config.image_size == "2K"
        assert config.response_modalities == ["IMAGE"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "size,expected",
        [
            ((10, 10), "1K"),
            ((1024, 10), "1K"),
            ((1500, 10), "2K"),
            ((10, 3000), "4K"),
        ],
    )
    async def test_image_size_matches_source_tier(self, tmp_path, size, expected):
        """The output tier is the smallest that covers the source's long edge."""
        image_path = tmp_path / "input.png"
        self._write_png(image_path, size=size)
        client, mock_client = _client_returning(_image_response(_png_bytes()))

        result = await client.advanced_edit(
            image_path=image_path,
            prompt="a hat",
            edit_mode="inpaint_insertion",
            output_path=tmp_path / "edited.png",
        )

        assert result.success is True
        config = mock_client.aio.models.generate_content.call_args.kwargs["config"]
        assert config.image_config.image_size == expected
        assert result.data["image_size"] == expected

    @pytest.mark.asyncio
    async def test_lite_model_drops_to_1k(self, tmp_path):
        """A 1K-only model gets 1K even when the source is larger."""
        image_path = tmp_path / "input.png"
        self._write_png(image_path, size=(1500, 10))
        client, mock_client = _client_returning(_image_response(_png_bytes()))

        result = await client.advanced_edit(
            image_path=image_path,
            prompt="watercolor",
            edit_mode="style_transfer",
            output_path=tmp_path / "edited.png",
            model="gemini-3.1-flash-lite-image",
        )

        assert result.success is True
        kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        assert kwargs["model"] == "gemini-3.1-flash-lite-image"
        assert kwargs["config"].image_config.image_size == "1K"
        assert result.data["image_size"] == "1K"
        assert result.data["model"] == "gemini-3.1-flash-lite-image"

    @pytest.mark.asyncio
    async def test_non_panoramic_model_gets_nearest_standard_ratio(self, tmp_path):
        """A 5:1 source on the lite model is pinned to 21:9 with a warning."""
        image_path = tmp_path / "input.png"
        self._write_png(image_path, size=(1500, 300))
        client, mock_client = _client_returning(_image_response(_png_bytes()))

        result = await client.advanced_edit(
            image_path=image_path,
            prompt="edit",
            edit_mode="background_swap",
            output_path=tmp_path / "edited.png",
            model="gemini-3.1-flash-lite-image",
        )

        assert result.success is True
        config = mock_client.aio.models.generate_content.call_args.kwargs["config"]
        assert config.image_config.aspect_ratio == "21:9"
        warnings = result.data["warnings"]
        assert any("1500x300" in w and "21:9" in w for w in warnings)

    @pytest.mark.asyncio
    async def test_unknown_mode_fails_without_api_call(self, tmp_path):
        """An unknown edit mode returns a failed result and skips the API."""
        image_path = tmp_path / "input.png"
        self._write_png(image_path)
        client, mock_client = _client_returning(_image_response(_png_bytes()))

        result = await client.advanced_edit(
            image_path=image_path,
            prompt="edit",
            edit_mode="not_a_mode",
            output_path=tmp_path / "edited.png",
        )

        assert result.success is False
        assert result.error == "Unknown edit mode: not_a_mode"
        mock_client.aio.models.generate_content.assert_not_called()

    @pytest.mark.asyncio
    async def test_retired_model_fails_without_api_call(self, tmp_path):
        """A retired Imagen ID returns the migration error and skips the API."""
        image_path = tmp_path / "input.png"
        self._write_png(image_path)
        client, mock_client = _client_returning(_image_response(_png_bytes()))

        result = await client.advanced_edit(
            image_path=image_path,
            prompt="edit",
            edit_mode="background_swap",
            output_path=tmp_path / "edited.png",
            model="imagen-4.0-generate-001",
        )

        assert result.success is False
        assert "retired" in result.error
        assert "gemini-nano-banana-2.1" in result.error
        mock_client.aio.models.generate_content.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_image_reports_finish_reason(self, tmp_path):
        """A response without an image surfaces the finish reason."""
        image_path = tmp_path / "input.png"
        self._write_png(image_path)
        client, _ = _client_returning(
            _no_image_response(types.FinishReason.IMAGE_PROHIBITED_CONTENT)
        )

        result = await client.advanced_edit(
            image_path=image_path,
            prompt="edit",
            edit_mode="product_image",
            output_path=tmp_path / "edited.png",
        )

        assert result.success is False
        assert result.error == (
            "No image returned (finish_reason=IMAGE_PROHIBITED_CONTENT)"
        )

    @pytest.mark.asyncio
    async def test_api_error_is_classified(self, tmp_path):
        """API exceptions are mapped to user-facing messages."""
        image_path = tmp_path / "input.png"
        self._write_png(image_path)
        client, mock_client = _client_returning(None)
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("429 RESOURCE_EXHAUSTED: quota exceeded")
        )

        result = await client.advanced_edit(
            image_path=image_path,
            prompt="edit",
            edit_mode="background_swap",
            output_path=tmp_path / "edited.png",
        )

        assert result.success is False
        assert result.error.startswith("Rate limit exceeded")
        assert "quota exceeded" in result.error


class TestModelResolution:
    """Tests for retired model rejection and deprecated alias rewriting."""

    @staticmethod
    def _mock_image_client():
        """Build a mock genai client whose generate_content returns an image."""
        buf = io.BytesIO()
        Image.new("RGB", (10, 10), color="red").save(buf, format="PNG")

        mock_part = Mock()
        mock_part.inline_data = Mock(data=buf.getvalue())
        mock_candidate = Mock()
        mock_candidate.content.parts = [mock_part]
        mock_response = Mock(candidates=[mock_candidate])
        mock_response.prompt_feedback = None

        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        return mock_client

    @staticmethod
    def _write_input(path):
        Image.new("RGB", (10, 10), color="blue").save(str(path))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("retired_model", sorted(RETIRED_MODELS))
    async def test_generate_rejects_retired_imagen_model(self, tmp_path, retired_model):
        """Retired Imagen IDs fail fast with a migration error and no SDK call."""
        mock_client = self._mock_image_client()
        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        result = await client.generate(
            prompt="test",
            output_path=tmp_path / "out.png",
            model=retired_model,
        )

        assert result.success is False
        assert "retired" in result.error
        assert retired_model in result.error
        assert "gemini-nano-banana-2.1" in result.error
        assert "2026-08-17" in result.error
        assert not (tmp_path / "out.png").exists()
        mock_client.aio.models.generate_content.assert_not_called()
        mock_client.models.generate_images.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_rejects_retired_default_model(self, tmp_path):
        """A retired model set as the client default is rejected too."""
        mock_client = self._mock_image_client()
        client = ImagenAPIClient(
            model_name="imagen-4.0-fast-generate-001", api_key="test-key"
        )
        client._genai_client = mock_client

        result = await client.generate(prompt="test", output_path=tmp_path / "o.png")

        assert result.success is False
        assert "retired" in result.error
        mock_client.aio.models.generate_content.assert_not_called()

    @pytest.mark.asyncio
    async def test_edit_rejects_retired_imagen_model(self, tmp_path):
        """edit() applies the same retired model rejection."""
        mock_client = self._mock_image_client()
        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client
        input_path = tmp_path / "input.png"
        self._write_input(input_path)

        result = await client.edit(
            prompt="add clouds",
            input_path=input_path,
            output_path=tmp_path / "out.png",
            model="imagen-4.0-generate-001",
        )

        assert result.success is False
        assert "retired" in result.error
        assert "gemini-nano-banana-2.1" in result.error
        mock_client.aio.models.generate_content.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "old_id,new_id",
        sorted(DEPRECATED_MODEL_ALIASES.items()),
    )
    async def test_generate_rewrites_preview_alias(self, tmp_path, old_id, new_id):
        """Shut-down preview IDs call the stable model and add a warning."""
        mock_client = self._mock_image_client()
        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        result = await client.generate(
            prompt="test",
            output_path=tmp_path / "out.png",
            model=old_id,
        )

        assert result.success is True
        call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == new_id
        assert result.data["model"] == new_id
        assert len(result.data["warnings"]) == 1
        assert old_id in result.data["warnings"][0]
        assert new_id in result.data["warnings"][0]

    @pytest.mark.asyncio
    async def test_generate_alias_and_person_generation_warnings_combine(
        self, tmp_path
    ):
        """Alias and person_generation warnings share one warnings list."""
        mock_client = self._mock_image_client()
        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        result = await client.generate(
            prompt="test",
            output_path=tmp_path / "out.png",
            model="gemini-3.1-flash-image-preview",
            person_generation="allow",
        )

        assert result.success is True
        warnings = result.data["warnings"]
        assert len(warnings) == 2
        assert "gemini-3.1-flash-image-preview" in warnings[0]
        assert "person_generation='allow'" in warnings[1]
        assert "Imagen" not in warnings[1]

    @pytest.mark.asyncio
    async def test_generate_stable_model_has_no_warnings_key(self, tmp_path):
        """Stable models without person_generation add no warnings key."""
        mock_client = self._mock_image_client()
        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        result = await client.generate(
            prompt="test",
            output_path=tmp_path / "out.png",
            model="gemini-3.1-flash-image",
        )

        assert result.success is True
        assert result.data["model"] == "gemini-3.1-flash-image"
        assert "warnings" not in result.data

    @pytest.mark.asyncio
    async def test_generate_default_model_is_nano_banana_2_1(self, tmp_path):
        """With no model set, generate_content gets gemini-nano-banana-2.1."""
        mock_client = self._mock_image_client()
        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        result = await client.generate(prompt="test", output_path=tmp_path / "o.png")

        assert result.success is True
        call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "gemini-nano-banana-2.1"
        assert result.data["model"] == "gemini-nano-banana-2.1"

    @pytest.mark.asyncio
    async def test_generate_does_not_rewrite_legacy_stable_ids(self, tmp_path):
        """The deprecated gemini-2.5-flash-image still resolves unchanged."""
        mock_client = self._mock_image_client()
        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        result = await client.generate(
            prompt="test",
            output_path=tmp_path / "out.png",
            model="gemini-2.5-flash-image",
        )

        assert result.success is True
        call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "gemini-2.5-flash-image"
        assert "warnings" not in result.data

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "old_id,new_id",
        sorted(DEPRECATED_MODEL_ALIASES.items()),
    )
    async def test_edit_rewrites_preview_alias(self, tmp_path, old_id, new_id):
        """edit() rewrites preview IDs and reports the substitution."""
        mock_client = self._mock_image_client()
        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client
        input_path = tmp_path / "input.png"
        self._write_input(input_path)

        result = await client.edit(
            prompt="add clouds",
            input_path=input_path,
            output_path=tmp_path / "out.png",
            model=old_id,
        )

        assert result.success is True
        call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == new_id
        assert result.data["model"] == new_id
        assert len(result.data["warnings"]) == 1
        assert old_id in result.data["warnings"][0]
        assert new_id in result.data["warnings"][0]

    @pytest.mark.asyncio
    async def test_edit_stable_model_has_no_warnings_key(self, tmp_path):
        """edit() with a stable model adds no warnings key."""
        mock_client = self._mock_image_client()
        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client
        input_path = tmp_path / "input.png"
        self._write_input(input_path)

        result = await client.edit(
            prompt="add clouds",
            input_path=input_path,
            output_path=tmp_path / "out.png",
        )

        assert result.success is True
        assert result.data["model"] == "gemini-nano-banana-2.1"
        assert "warnings" not in result.data


class TestAnalysisModelOverride:
    """Tests that analysis methods default to TEXT_MODEL and honor overrides."""

    @staticmethod
    def _text_client(text):
        mock_part = Mock()
        mock_part.text = text
        mock_candidate = Mock()
        mock_candidate.content.parts = [mock_part]
        mock_response = Mock(candidates=[mock_candidate])
        mock_response.prompt_feedback = None

        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        return mock_client

    @staticmethod
    def _image(tmp_path, name="img.png"):
        path = tmp_path / name
        Image.new("RGB", (10, 10), color="red").save(str(path))
        return path

    @pytest.mark.asyncio
    async def test_analyze_defaults_to_text_model(self, tmp_path):
        """analyze() calls gemini-3.5-flash-lite by default."""
        mock_client = self._text_client("a red square")
        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        result = await client.analyze(self._image(tmp_path))

        assert result.success is True
        call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "gemini-3.5-flash-lite"

    @pytest.mark.asyncio
    async def test_analyze_honors_model_override(self, tmp_path):
        """analyze() passes a model override to the SDK."""
        mock_client = self._text_client("a red square")
        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        await client.analyze(self._image(tmp_path), model="gemini-3.8-flash")

        call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "gemini-3.8-flash"

    @pytest.mark.asyncio
    async def test_extract_text_defaults_to_text_model(self, tmp_path):
        """extract_text() calls gemini-3.5-flash-lite by default."""
        mock_client = self._text_client('{"text": "hi", "blocks": []}')
        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        await client.extract_text(self._image(tmp_path))

        call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "gemini-3.5-flash-lite"

    @pytest.mark.asyncio
    async def test_detect_objects_honors_model_override(self, tmp_path):
        """detect_objects() passes a model override to the SDK."""
        mock_client = self._text_client("[]")
        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        await client.detect_objects(self._image(tmp_path), model="gemini-3.8-flash")

        call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "gemini-3.8-flash"

    @pytest.mark.asyncio
    async def test_compare_images_honors_model_override(self, tmp_path):
        """compare_images() passes a model override to the SDK."""
        mock_client = self._text_client("they differ")
        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        await client.compare_images(
            [self._image(tmp_path, "a.png"), self._image(tmp_path, "b.png")],
            model="gemini-3.8-flash",
        )

        call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "gemini-3.8-flash"

    @pytest.mark.asyncio
    async def test_optimize_prompt_defaults_and_overrides(self):
        """optimize_prompt() defaults to the text model and honors overrides."""
        mock_client = self._text_client("a vivid prompt")
        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        await client.optimize_prompt("a cat")
        default_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        assert default_kwargs["model"] == "gemini-3.5-flash-lite"

        await client.optimize_prompt("a cat", model="gemini-3.8-flash")
        override_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        assert override_kwargs["model"] == "gemini-3.8-flash"


class TestUpscale:
    """Tests for generative upscale via generate_content."""

    @staticmethod
    def _write_png(path, size=(1024, 1024)):
        """Write a solid-colour PNG of the given size to ``path``."""
        Image.new("RGB", size, color="red").save(str(path))

    @pytest.mark.asyncio
    async def test_upscale_rerenders_with_default_model(self, tmp_path):
        """x2 from 1K sends the source, the instruction, and a 2K config."""
        image_path = tmp_path / "input.png"
        self._write_png(image_path)
        output_path = tmp_path / "upscaled.png"
        client, mock_client = _client_returning(
            _image_response(_png_bytes(size=(32, 32), color="green"))
        )

        result = await client.upscale(
            image_path=image_path, output_path=output_path, upscale_factor="x2"
        )

        assert result.success is True
        assert output_path.exists()
        assert result.image_paths == [str(output_path)]
        mock_client.aio.models.generate_content.assert_awaited_once()
        kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        assert kwargs["model"] == "gemini-nano-banana-2.1"
        source, instruction = kwargs["contents"]
        assert isinstance(source, Image.Image)
        assert source.size == (1024, 1024)
        assert instruction == UPSCALE_INSTRUCTION
        config = kwargs["config"]
        assert config.response_modalities == ["IMAGE"]
        assert config.image_config.image_size == "2K"
        assert config.image_config.aspect_ratio == "1:1"

        data = result.data
        assert data["upscale_factor"] == "x2"
        assert data["model"] == "gemini-nano-banana-2.1"
        assert data["method"] == "generative"
        assert data["image_size"] == "2K"
        assert data["source_size"] == [1024, 1024]
        assert data["output_size"] == [32, 32]
        assert data["image_path"] == str(output_path)
        assert data["warnings"] == [GENERATIVE_UPSCALE_WARNING]

    @pytest.mark.asyncio
    async def test_upscale_x4_targets_4k_with_model_override(self, tmp_path):
        """x4 from 1K asks for 4K and honours the model override."""
        image_path = tmp_path / "input.png"
        self._write_png(image_path)
        client, mock_client = _client_returning(_image_response(_png_bytes()))

        result = await client.upscale(
            image_path=image_path,
            output_path=tmp_path / "upscaled.png",
            upscale_factor="x4",
            model="gemini-3-pro-image",
        )

        assert result.success is True
        kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        assert kwargs["model"] == "gemini-3-pro-image"
        assert kwargs["config"].image_config.image_size == "4K"
        assert result.data["image_size"] == "4K"
        assert result.data["model"] == "gemini-3-pro-image"

    @pytest.mark.asyncio
    async def test_upscale_pins_nearest_aspect_ratio(self, tmp_path):
        """A 16:9 source is re-rendered at 16:9."""
        image_path = tmp_path / "input.png"
        self._write_png(image_path, size=(1600, 900))
        client, mock_client = _client_returning(_image_response(_png_bytes()))

        result = await client.upscale(
            image_path=image_path,
            output_path=tmp_path / "upscaled.png",
            upscale_factor="x2",
        )

        assert result.success is True
        config = mock_client.aio.models.generate_content.call_args.kwargs["config"]
        assert config.image_config.aspect_ratio == "16:9"
        assert config.image_config.image_size == "4K"
        assert result.data["source_size"] == [1600, 900]

    @pytest.mark.asyncio
    async def test_upscale_cap_warning_is_reported(self, tmp_path):
        """x2 beyond 4K is capped at 4K with a warning."""
        image_path = tmp_path / "input.png"
        self._write_png(image_path, size=(3000, 10))
        client, _ = _client_returning(_image_response(_png_bytes()))

        result = await client.upscale(
            image_path=image_path,
            output_path=tmp_path / "upscaled.png",
            upscale_factor="x2",
        )

        assert result.success is True
        assert result.data["image_size"] == "4K"
        warnings = result.data["warnings"]
        assert warnings[0] == GENERATIVE_UPSCALE_WARNING
        assert "6000px" in warnings[1]
        assert "capped at 4K" in warnings[1]

    @pytest.mark.asyncio
    async def test_upscale_clamps_panoramic_ratio_for_pro_model(self, tmp_path):
        """gemini-3-pro-image has no panoramic output; 4:1 becomes 21:9."""
        image_path = tmp_path / "input.png"
        self._write_png(image_path, size=(2000, 500))
        client, mock_client = _client_returning(_image_response(_png_bytes()))

        result = await client.upscale(
            image_path=image_path,
            output_path=tmp_path / "upscaled.png",
            upscale_factor="x2",
            model="gemini-3-pro-image",
        )

        assert result.success is True
        config = mock_client.aio.models.generate_content.call_args.kwargs["config"]
        assert config.image_config.aspect_ratio == "21:9"
        warnings = result.data["warnings"]
        assert warnings[0] == GENERATIVE_UPSCALE_WARNING
        assert any("2000x500" in w and "21:9" in w for w in warnings)

    @pytest.mark.asyncio
    async def test_upscale_keeps_panoramic_ratio_for_default_model(self, tmp_path):
        """The default model honours 4:1, so no ratio warning is added."""
        image_path = tmp_path / "input.png"
        self._write_png(image_path, size=(2000, 500))
        client, mock_client = _client_returning(_image_response(_png_bytes()))

        result = await client.upscale(
            image_path=image_path,
            output_path=tmp_path / "upscaled.png",
            upscale_factor="x2",
        )

        assert result.success is True
        config = mock_client.aio.models.generate_content.call_args.kwargs["config"]
        assert config.image_config.aspect_ratio == "4:1"
        assert result.data["warnings"] == [GENERATIVE_UPSCALE_WARNING]

    @pytest.mark.asyncio
    async def test_upscale_rejects_1k_only_model(self, tmp_path):
        """A 1K-only model cannot reach 2K; the error names the alternatives."""
        image_path = tmp_path / "input.png"
        self._write_png(image_path)
        client, mock_client = _client_returning(_image_response(_png_bytes()))

        result = await client.upscale(
            image_path=image_path,
            output_path=tmp_path / "upscaled.png",
            upscale_factor="x2",
            model="gemini-3.1-flash-lite-image",
        )

        assert result.success is False
        assert result.error.startswith("gemini-3.1-flash-lite-image cannot output 2K")
        assert "gemini-nano-banana-2.1" in result.error
        assert "gemini-3-pro-image" in result.error
        mock_client.aio.models.generate_content.assert_not_called()

    @pytest.mark.asyncio
    async def test_upscale_rejects_retired_imagen_model(self, tmp_path):
        """A retired Imagen ID returns the migration error and skips the API."""
        image_path = tmp_path / "input.png"
        self._write_png(image_path)
        client, mock_client = _client_returning(_image_response(_png_bytes()))

        result = await client.upscale(
            image_path=image_path,
            output_path=tmp_path / "upscaled.png",
            upscale_factor="x2",
            model="imagen-4.0-generate-001",
        )

        assert result.success is False
        assert "retired" in result.error
        assert "gemini-nano-banana-2.1" in result.error
        mock_client.aio.models.generate_content.assert_not_called()

    @pytest.mark.asyncio
    async def test_upscale_rewrites_preview_alias_with_warning(self, tmp_path):
        """A shut-down preview ID is rewritten and the warning is appended."""
        image_path = tmp_path / "input.png"
        self._write_png(image_path)
        client, mock_client = _client_returning(_image_response(_png_bytes()))

        result = await client.upscale(
            image_path=image_path,
            output_path=tmp_path / "upscaled.png",
            upscale_factor="x2",
            model="gemini-3-pro-image-preview",
        )

        assert result.success is True
        kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        assert kwargs["model"] == "gemini-3-pro-image"
        assert result.data["model"] == "gemini-3-pro-image"
        assert any("gemini-3-pro-image-preview" in w for w in result.data["warnings"])

    @pytest.mark.asyncio
    async def test_upscale_source_already_4k_fails_without_api_call(self, tmp_path):
        """A source at the 4K ceiling cannot be upscaled."""
        image_path = tmp_path / "input.png"
        self._write_png(image_path, size=(4096, 10))
        client, mock_client = _client_returning(_image_response(_png_bytes()))

        result = await client.upscale(
            image_path=image_path,
            output_path=tmp_path / "upscaled.png",
            upscale_factor="x2",
        )

        assert result.success is False
        assert "already 4096px" in result.error
        mock_client.aio.models.generate_content.assert_not_called()

    @pytest.mark.asyncio
    async def test_upscale_no_image_reports_finish_reason(self, tmp_path):
        """A response without an image surfaces the finish reason."""
        image_path = tmp_path / "input.png"
        self._write_png(image_path)
        output_path = tmp_path / "upscaled.png"
        client, _ = _client_returning(
            _no_image_response(types.FinishReason.IMAGE_SAFETY)
        )

        result = await client.upscale(
            image_path=image_path, output_path=output_path, upscale_factor="x4"
        )

        assert result.success is False
        assert result.error == "No image returned (finish_reason=IMAGE_SAFETY)"
        assert not output_path.exists()


class TestBackendSelection:
    """Backend detection and google-genai client construction."""

    @pytest.fixture(autouse=True)
    def _clear_credential_env(self, monkeypatch):
        for var in CREDENTIAL_ENV_VARS:
            monkeypatch.delenv(var, raising=False)

    def test_no_credentials_defaults_to_gemini_api(self):
        client = ImagenAPIClient()
        assert client.backend == BACKEND_GEMINI_API
        assert client.project_id is None
        assert client.location == DEFAULT_VERTEX_LOCATION == "global"

    def test_api_key_selects_gemini_api(self):
        client = ImagenAPIClient(api_key="test-key")
        assert client.backend == BACKEND_GEMINI_API

    def test_project_without_key_selects_vertex_ai(self):
        client = ImagenAPIClient(project_id="my-project")
        assert client.backend == BACKEND_VERTEX_AI
        assert client.project_id == "my-project"
        assert client.location == "global"

    def test_project_env_without_key_selects_vertex_ai(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us")
        client = ImagenAPIClient()
        assert client.backend == BACKEND_VERTEX_AI
        assert client.project_id == "my-project"
        assert client.location == "us"

    @pytest.mark.parametrize(
        "var", ["GOOGLE_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY", "GEMINI_API_KEY"]
    )
    def test_key_env_with_project_selects_gemini_api(self, monkeypatch, var):
        monkeypatch.setenv(var, "env-key")
        client = ImagenAPIClient(project_id="my-project")
        assert client.backend == BACKEND_GEMINI_API

    def test_explicit_backend_overrides_detection(self):
        client = ImagenAPIClient(
            api_key="test-key", backend=BACKEND_VERTEX_AI, project_id="my-project"
        )
        assert client.backend == BACKEND_VERTEX_AI

    @pytest.mark.parametrize(
        "var", ["GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_ENTERPRISE"]
    )
    def test_sdk_switch_prefers_vertex_ai_over_key(self, monkeypatch, var):
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        monkeypatch.setenv(var, "true")
        client = ImagenAPIClient(project_id="my-project")
        assert client.backend == BACKEND_VERTEX_AI

    def test_sdk_switch_without_project_stays_on_gemini_api(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
        client = ImagenAPIClient(api_key="test-key")
        assert client.backend == BACKEND_GEMINI_API

    @patch("google.genai.Client")
    def test_gemini_client_pins_vertexai_false_despite_sdk_switch(
        self, mock_client_cls, monkeypatch
    ):
        """The SDK would reroute an API key to Vertex AI unless told not to."""
        monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "true")
        client = ImagenAPIClient(api_key="test-key")

        client._get_genai_client()

        kwargs = mock_client_cls.call_args.kwargs
        assert kwargs["vertexai"] is False
        assert kwargs["api_key"] == "test-key"

    @patch("google.genai.Client")
    def test_vertex_client_construction(self, mock_client_cls):
        client = ImagenAPIClient(project_id="my-project", location="global")

        genai_client = client._get_genai_client()

        mock_client_cls.assert_called_once()
        kwargs = mock_client_cls.call_args.kwargs
        assert kwargs["vertexai"] is True
        assert kwargs["project"] == "my-project"
        assert kwargs["location"] == "global"
        assert "api_key" not in kwargs
        retry = kwargs["http_options"].retry_options
        assert retry.attempts == RETRY_ATTEMPTS == 5
        assert retry.http_status_codes == RETRY_STATUS_CODES
        assert 429 in retry.http_status_codes
        assert genai_client is mock_client_cls.return_value
        # The client is cached.
        assert client._get_genai_client() is genai_client
        mock_client_cls.assert_called_once()

    @patch("google.genai.Client")
    def test_gemini_client_construction(self, mock_client_cls):
        client = ImagenAPIClient(api_key="test-key")

        client._get_genai_client()

        kwargs = mock_client_cls.call_args.kwargs
        assert kwargs["api_key"] == "test-key"
        assert kwargs["vertexai"] is False
        assert "project" not in kwargs
        assert "location" not in kwargs
        assert kwargs["http_options"].retry_options.attempts == 5

    @patch("google.genai.Client")
    def test_gemini_client_uses_env_key(self, mock_client_cls, monkeypatch):
        monkeypatch.setenv("GOOGLE_GENERATIVE_AI_API_KEY", "env-key")
        client = ImagenAPIClient()

        client._get_genai_client()

        assert mock_client_cls.call_args.kwargs["api_key"] == "env-key"

    @patch("google.genai.Client")
    def test_vertex_without_project_raises(self, mock_client_cls):
        client = ImagenAPIClient(backend=BACKEND_VERTEX_AI)

        with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
            client._get_genai_client()

        mock_client_cls.assert_not_called()

    @patch("google.genai.Client")
    def test_gemini_without_key_raises(self, mock_client_cls):
        client = ImagenAPIClient()

        with pytest.raises(ValueError, match="No credentials found"):
            client._get_genai_client()

        mock_client_cls.assert_not_called()


class TestExtractImages:
    """Tests for _extract_images response handling."""

    def test_returns_inline_images(self):
        images, reason = ImagenAPIClient._extract_images(_image_response(_png_bytes()))
        assert len(images) == 1
        assert isinstance(images[0], Image.Image)
        assert reason is None

    @pytest.mark.parametrize("candidates", [None, []])
    def test_no_candidates(self, candidates):
        response = Mock(candidates=candidates)

        images, reason = ImagenAPIClient._extract_images(response)

        assert images == []
        assert reason == "No image returned (finish_reason=no candidates)"

    def test_candidate_without_content(self):
        candidate = Mock()
        candidate.content = None
        candidate.finish_reason = types.FinishReason.NO_IMAGE
        response = Mock(candidates=[candidate])

        images, reason = ImagenAPIClient._extract_images(response)

        assert images == []
        assert reason == "No image returned (finish_reason=NO_IMAGE)"

    def test_parts_none_with_image_safety(self):
        response = _no_image_response(types.FinishReason.IMAGE_SAFETY, parts=None)

        images, reason = ImagenAPIClient._extract_images(response)

        assert images == []
        assert reason == "No image returned (finish_reason=IMAGE_SAFETY)"

    def test_text_only_parts_append_text(self):
        part = Mock()
        part.inline_data = None
        part.text = "  I cannot create that image.  "
        response = _no_image_response(types.FinishReason.STOP, parts=[part])

        images, reason = ImagenAPIClient._extract_images(response)

        assert images == []
        assert reason == (
            "No image returned (finish_reason=STOP): I cannot create that image."
        )

    def test_inline_part_without_data_is_skipped(self):
        part = Mock()
        part.inline_data = Mock(data=None)
        part.text = None
        response = _no_image_response(types.FinishReason.STOP, parts=[part])

        images, reason = ImagenAPIClient._extract_images(response)

        assert images == []
        assert reason == "No image returned (finish_reason=STOP)"

    def test_undecodable_inline_data_is_reported(self):
        part = Mock()
        part.inline_data = Mock(data=b"notanimage")
        part.text = None
        response = _no_image_response(types.FinishReason.STOP, parts=[part])

        images, reason = ImagenAPIClient._extract_images(response)

        assert images == []
        assert reason == (
            "No image returned (finish_reason=STOP); 1 undecodable image part(s)"
        )

    def test_undecodable_part_does_not_hide_a_good_image(self):
        bad = Mock()
        bad.inline_data = Mock(data=b"notanimage")
        good = Mock()
        good.inline_data = Mock(data=_png_bytes())
        response = _no_image_response(types.FinishReason.STOP, parts=[bad, good])

        images, reason = ImagenAPIClient._extract_images(response)

        assert len(images) == 1
        assert reason is None

    def test_string_finish_reason(self):
        response = _no_image_response("IMAGE_OTHER", parts=[])

        _, reason = ImagenAPIClient._extract_images(response)

        assert reason == "No image returned (finish_reason=IMAGE_OTHER)"

    def test_missing_finish_reason(self):
        response = _no_image_response(None, parts=[])

        _, reason = ImagenAPIClient._extract_images(response)

        assert reason == "No image returned (finish_reason=unknown)"


class TestSdkImageSize:
    """Tool-facing size tiers map to the values the API accepts."""

    @pytest.mark.parametrize(
        "given,expected",
        [("0.5K", "512px"), ("1K", "1K"), ("2K", "2K"), ("4K", "4K"), (None, None)],
    )
    def test_maps_tool_sizes(self, given, expected):
        assert ImagenAPIClient._sdk_image_size(given) == expected

    @pytest.mark.asyncio
    async def test_generate_sends_512px_for_half_k(self, tmp_path):
        client, mock_client = _client_returning(_image_response(_png_bytes()))

        result = await client.generate(
            prompt="a cat",
            output_path=tmp_path / "out.png",
            image_size="0.5K",
            model="gemini-3.1-flash-image",
        )

        assert result.success is True
        config = mock_client.aio.models.generate_content.call_args.kwargs["config"]
        assert config.image_config.image_size == "512px"
        assert result.data["image_size"] == "0.5K"

    @pytest.mark.asyncio
    async def test_generate_passes_2k_through(self, tmp_path):
        client, mock_client = _client_returning(_image_response(_png_bytes()))

        result = await client.generate(
            prompt="a cat", output_path=tmp_path / "out.png", image_size="2K"
        )

        assert result.success is True
        config = mock_client.aio.models.generate_content.call_args.kwargs["config"]
        assert config.image_config.image_size == "2K"


class TestGeometryHelpers:
    """Aspect ratio, size tier, and upscale target selection."""

    @pytest.mark.parametrize(
        "width,height,expected",
        [
            (1024, 1024, "1:1"),
            (2048, 512, "4:1"),
            (1600, 900, "16:9"),
            (900, 1600, "9:16"),
            (3000, 2000, "3:2"),
            (0, 100, "1:1"),
            (100, 0, "1:1"),
        ],
    )
    def test_nearest_aspect_ratio(self, width, height, expected):
        assert ImagenAPIClient._nearest_aspect_ratio(width, height) == expected

    def test_nearest_aspect_ratio_is_symmetric_in_log_space(self):
        """A wide source and its tall mirror get mirrored ratios."""
        assert ImagenAPIClient._nearest_aspect_ratio(3100, 1000) == "4:1"
        assert ImagenAPIClient._nearest_aspect_ratio(1000, 3100) == "1:4"

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("gemini-nano-banana-2.1", "4:1"),
            ("gemini-3.1-flash-image", "4:1"),
            ("gemini-3-pro-image", "21:9"),
            ("gemini-3.1-flash-lite-image", "21:9"),
            ("gemini-2.5-flash-image", "21:9"),
            ("custom-model", "4:1"),
            (None, "4:1"),
        ],
    )
    def test_nearest_aspect_ratio_respects_model_support(self, model, expected):
        assert ImagenAPIClient._nearest_aspect_ratio(2048, 512, model) == expected

    def test_tall_source_on_non_panoramic_model(self):
        ratio = ImagenAPIClient._nearest_aspect_ratio(512, 2048, "gemini-3-pro-image")
        assert ratio == "9:16"

    def test_aspect_ratio_warning_only_when_shape_drifts(self):
        warn = ImagenAPIClient._aspect_ratio_warning
        assert warn(1600, 900, "16:9", "m") is None
        assert warn(1040, 1000, "1:1", "m") is None
        assert warn(0, 10, "1:1", "m") is None
        warning = warn(2048, 512, "21:9", "gemini-3-pro-image")
        assert warning is not None
        assert "2048x512" in warning
        assert "21:9" in warning
        assert "gemini-3-pro-image" in warning

    @pytest.mark.asyncio
    async def test_list_models_panoramic_flags_match_aspect_support(self):
        """The metadata shown to users agrees with the ratio clamp."""
        result = await ImagenAPIClient(api_key="test-key").list_models()

        for model in result.data["models"]:
            expected = "4:1" in supported_aspect_ratios(model["name"])
            assert model["capabilities"]["panoramic"] is expected, model["name"]

    @pytest.mark.parametrize(
        "edge,expected",
        [
            (1, "1K"),
            (512, "1K"),
            (1024, "1K"),
            (1025, "2K"),
            (2048, "2K"),
            (2049, "4K"),
            (4096, "4K"),
            (9000, "4K"),
        ],
    )
    def test_size_tier_for_edge(self, edge, expected):
        assert ImagenAPIClient._size_tier_for_edge(edge) == expected

    def test_upscale_target_x2_from_1k(self):
        assert ImagenAPIClient._upscale_target(1024, 1024, "x2") == ("2K", None, None)

    def test_upscale_target_x4_from_1k(self):
        assert ImagenAPIClient._upscale_target(1024, 1024, "x4") == ("4K", None, None)

    def test_upscale_target_x2_from_512(self):
        assert ImagenAPIClient._upscale_target(512, 512, "x2") == ("1K", None, None)

    def test_upscale_target_uses_long_edge(self):
        assert ImagenAPIClient._upscale_target(100, 1024, "x2") == ("2K", None, None)

    def test_upscale_target_caps_at_4k_with_warning(self):
        tier, warning, error = ImagenAPIClient._upscale_target(3000, 2000, "x2")

        assert tier == "4K"
        assert error is None
        assert "6000px" in warning
        assert "capped at 4K (4096px)" in warning

    def test_upscale_target_rejects_source_at_4k(self):
        tier, warning, error = ImagenAPIClient._upscale_target(4096, 1000, "x2")

        assert tier is None
        assert warning is None
        assert "already 4096px" in error
        assert "4K (4096px)" in error


class TestClassifyError:
    """Tests for _classify_error messages."""

    @staticmethod
    def _client_error(code, status, message):
        body = {"error": {"code": code, "message": message, "status": status}}
        return errors.ClientError(code, body)

    @pytest.mark.parametrize("casing", ["Publisher Model", "Publisher model"])
    def test_vertex_regional_not_found_adds_location_hint(self, casing):
        """Both casings seen from the API get the location hint."""
        err = self._client_error(
            404,
            "NOT_FOUND",
            f"{casing} `projects/my-project/locations/us-central1/publishers/"
            "google/models/gemini-nano-banana-2.1` was not found or your "
            "project does not have access to it.",
        )

        message = ImagenAPIClient._classify_error(err)

        assert message.startswith(casing)
        assert "GOOGLE_CLOUD_LOCATION=global" in message

    def test_vertex_global_not_found_names_unknown_model(self):
        """At global the location hint is wrong; the model ID is the problem."""
        err = self._client_error(
            404,
            "NOT_FOUND",
            "Publisher Model `projects/my-project/locations/global/publishers/"
            "google/models/gemini-does-not-exist` was not found or your "
            "project does not have access to it.",
        )

        message = ImagenAPIClient._classify_error(err)

        assert "GOOGLE_CLOUD_LOCATION" not in message
        assert "unknown or retired" in message
        assert "list_available_models" in message

    def test_plain_exception_not_found_keeps_text(self):
        err = Exception(
            "404 NOT_FOUND. Publisher model `projects/my-project/locations/"
            "us-central1/publishers/google/models/gemini-nano-banana-2.1` "
            "was not found or your project does not have access to it."
        )

        message = ImagenAPIClient._classify_error(err)

        assert message.startswith("404 NOT_FOUND")
        assert "GOOGLE_CLOUD_LOCATION=global" in message

    def test_quota_error_keeps_api_text(self):
        """A quota of 0 is an account limit; the text must survive."""
        err = self._client_error(
            429,
            "RESOURCE_EXHAUSTED",
            "Quota exceeded for aiplatform.googleapis.com/"
            "online_prediction_requests_per_base_model, limit: 0",
        )

        message = ImagenAPIClient._classify_error(err)

        assert message.startswith("Rate limit exceeded after 5 attempts")
        assert "RESOURCE_EXHAUSTED" in message
        assert "limit: 0" in message
        assert "quota increase" in message

    def test_resource_exhausted_text_is_rate_limit(self):
        err = Exception("429 RESOURCE_EXHAUSTED: quota exceeded")

        message = ImagenAPIClient._classify_error(err)

        assert message.startswith("Rate limit exceeded")
        assert "quota exceeded" in message

    def test_permission_denied_points_at_project_setup(self):
        err = self._client_error(
            403,
            "PERMISSION_DENIED",
            "Vertex AI API has not been used in project 123456789012 before "
            "or it is disabled.",
        )

        message = ImagenAPIClient._classify_error(err)

        assert message.startswith("Permission denied (PERMISSION_DENIED)")
        assert "has not been used in project" in message
        assert "roles/aiplatform.user" in message

    def test_invalid_api_key(self):
        err = self._client_error(
            400,
            "INVALID_ARGUMENT",
            "API key not valid. Please pass a valid API key.",
        )

        message = ImagenAPIClient._classify_error(err)

        assert message.startswith("Invalid API key:")
        assert "API key not valid" in message

    def test_invalid_argument_keeps_detail(self):
        err = self._client_error(
            400,
            "INVALID_ARGUMENT",
            "Unable to submit request because image_size 2K is not supported",
        )

        message = ImagenAPIClient._classify_error(err)

        assert message.startswith("Invalid request (INVALID_ARGUMENT)")
        assert "image_size 2K is not supported" in message

    def test_server_error_keeps_code_and_text(self):
        body = {
            "error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}
        }
        err = errors.ServerError(503, body)

        assert ImagenAPIClient._classify_error(err) == "503 UNAVAILABLE: overloaded"

    def test_other_errors_pass_through(self):
        assert ImagenAPIClient._classify_error(Exception("boom")) == "boom"
