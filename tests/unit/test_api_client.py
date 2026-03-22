"""Unit tests for API client using google-genai SDK."""

import io
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from pixelforge_mcp.utils.api_client import (
    COMPARE_IMAGES_PROMPT,
    OCR_PROMPT,
    PILLOW_FORMAT_MAP,
    SAFETY_PRESETS,
    TEXT_MODEL,
    GenerationResult,
    ImagenAPIClient,
)


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
        assert client.model_name == "gemini-2.5-flash-image"
        assert client.api_key is None
        assert client.log_images is False

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        client = ImagenAPIClient(
            model_name="gemini-3-pro-image-preview",
            api_key="test-key",
            log_images=True,
        )
        assert client.model_name == "gemini-3-pro-image-preview"
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
        """Test generation failure when no images in response."""
        mock_candidate = Mock()
        mock_candidate.content.parts = []
        mock_response = Mock(candidates=[mock_candidate])
        mock_response.prompt_feedback = None

        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client
        output_path = tmp_path / "test.png"

        result = await client.generate(prompt="test", output_path=output_path)

        assert result.success is False
        assert result.error == "No images generated"

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
        """Test failed image editing when response has no inline_data."""
        mock_candidate = Mock()
        mock_candidate.content.parts = []
        mock_response = Mock(candidates=[mock_candidate])
        mock_response.prompt_feedback = None

        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        input_path = tmp_path / "input.png"
        Image.new("RGB", (10, 10)).save(str(input_path))
        output_path = tmp_path / "output.png"

        result = await client.edit(
            prompt="add clouds", input_path=input_path, output_path=output_path
        )

        assert result.success is False
        assert result.error == "No images generated"

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
        assert "gemini-2.5-flash-image" in model_names
        assert "gemini-3-pro-image-preview" in model_names
        assert "gemini-3.1-flash-image-preview" in model_names
        assert "imagen-4.0-generate-001" in model_names
        assert "imagen-4.0-fast-generate-001" in model_names

        for model in models:
            assert "name" in model
            assert "nickname" in model
            assert "speed" in model
            assert "quality" in model
            assert "description" in model
            assert "best_for" in model
            assert "capabilities" in model
            assert "temperature" in model

    @pytest.mark.asyncio
    async def test_list_models_nano_banana_2_metadata(self):
        """Test Nano Banana 2 model metadata is correct."""
        client = ImagenAPIClient()
        result = await client.list_models()

        models = result.data["models"]
        nb2 = next(m for m in models if m["name"] == "gemini-3.1-flash-image-preview")

        assert nb2["nickname"] == "Nano Banana 2"
        assert nb2["speed"] == "fast"
        assert nb2["quality"] == "excellent"
        assert nb2["default"] is False
        assert nb2["capabilities"]["panoramic_ratios"] is True
        assert nb2["capabilities"]["text_rendering"] == "excellent"
        assert nb2["capabilities"]["grounding"] == "web search + Google Image Search"
        assert nb2["temperature"] == {"min": 0.0, "max": 2.0, "default": 1.0}

    @pytest.mark.asyncio
    async def test_list_models_person_generation_capability(self):
        """Test all models advertise person_generation capability."""
        client = ImagenAPIClient()
        result = await client.list_models()

        for model in result.data["models"]:
            assert model["capabilities"]["person_generation"] is True

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
            model="gemini-3-pro-image-preview",
        )

        # Verify the model was passed to the SDK call
        call_kwargs = mock_client.aio.models.generate_content.call_args[1]
        assert call_kwargs["model"] == "gemini-3-pro-image-preview"


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
            model="gemini-3-pro-image-preview",
        )

        assert result.success is True
        # Verify the model was passed to the SDK call
        call_kwargs = mock_client.aio.models.generate_content.call_args[1]
        assert call_kwargs["model"] == "gemini-3-pro-image-preview"

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

        assert TEXT_MODEL == "gemini-2.5-flash"

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
        """Test background removal with no images in response."""
        mock_candidate = Mock()
        mock_candidate.content.parts = []
        mock_response = Mock(candidates=[mock_candidate])
        mock_response.prompt_feedback = None

        mock_client = Mock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client

        image_path = tmp_path / "input.png"
        Image.new("RGB", (10, 10)).save(str(image_path))
        output_path = tmp_path / "output.png"

        result = await client.remove_background(image_path, output_path)

        assert result.success is False
        assert result.error == "Background removal failed"

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


class TestImagen4:
    """Tests for Imagen 4 generation via client.models.generate_images."""

    @pytest.mark.asyncio
    async def test_imagen4_generate_success(self, tmp_path):
        """Test successful Imagen 4 generation."""
        test_img = Image.new("RGB", (10, 10), color="green")

        mock_generated = Mock()
        mock_generated.image = test_img
        mock_response = Mock()
        mock_response.generated_images = [mock_generated]

        mock_client = Mock()
        mock_client.models.generate_images = Mock(return_value=mock_response)

        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client
        output_path = tmp_path / "imagen4_out.png"

        result = await client.generate(
            prompt="a cat sitting on a chair",
            output_path=output_path,
            model="imagen-4.0-generate-001",
        )

        assert result.success is True
        assert result.images is not None
        assert len(result.images) == 1
        assert result.image_paths == [str(output_path)]
        assert result.data["model"] == "imagen-4.0-generate-001"
        assert output_path.exists()

    @pytest.mark.asyncio
    async def test_imagen4_auto_routing(self, tmp_path):
        """Test that generate() auto-routes Imagen 4 models."""
        test_img = Image.new("RGB", (10, 10), color="blue")

        mock_generated = Mock()
        mock_generated.image = test_img
        mock_response = Mock()
        mock_response.generated_images = [mock_generated]

        mock_client = Mock()
        mock_client.models.generate_images = Mock(return_value=mock_response)
        # Also set up aio.models to verify it is NOT called
        mock_client.aio.models.generate_content = AsyncMock()

        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client
        output_path = tmp_path / "out.png"

        await client.generate(
            prompt="test",
            output_path=output_path,
            model="imagen-4.0-generate-001",
        )

        # Imagen 4 should use generate_images, NOT generate_content
        mock_client.models.generate_images.assert_called_once()
        mock_client.aio.models.generate_content.assert_not_called()

    @pytest.mark.asyncio
    async def test_imagen4_failure(self, tmp_path):
        """Test Imagen 4 failure when no images generated."""
        mock_response = Mock()
        mock_response.generated_images = []

        mock_client = Mock()
        mock_client.models.generate_images = Mock(return_value=mock_response)

        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = mock_client
        output_path = tmp_path / "out.png"

        result = await client.generate(
            prompt="test",
            output_path=output_path,
            model="imagen-4.0-generate-001",
        )

        assert result.success is False
        assert result.error == "No images generated"
