"""Unit tests for API client using gemini-imagen library."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from PIL import Image

from pixelforge_mcp.utils.api_client import (
    DEFAULT_ANALYSIS_PROMPT,
    PILLOW_FORMAT_MAP,
    SAFETY_PRESETS,
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
    @patch("pixelforge_mcp.utils.api_client.GeminiImageGenerator")
    async def test_generate_success(self, mock_generator_class, tmp_path):
        """Test successful image generation."""
        mock_generator = Mock()
        mock_result = Mock()
        mock_result.images = [Mock(spec=Image.Image)]
        mock_generator.generate = AsyncMock(return_value=mock_result)
        mock_generator_class.return_value = mock_generator

        client = ImagenAPIClient()
        output_path = tmp_path / "test.png"

        result = await client.generate(
            prompt="test prompt",
            output_path=output_path,
            aspect_ratio="16:9",
            temperature=0.8,
            model="gemini-2.5-flash-image",
        )

        assert result.success is True
        assert result.images == mock_result.images
        assert result.image_paths == [str(output_path)]
        assert result.data["model"] == "gemini-2.5-flash-image"
        assert result.data["aspect_ratio"] == "16:9"
        assert result.data["temperature"] == 0.8

        # Verify generate was called with correct args
        mock_generator.generate.assert_called_once()
        call_kwargs = mock_generator.generate.call_args[1]
        assert call_kwargs["prompt"] == "test prompt"
        assert call_kwargs["output_images"] == [str(output_path)]
        assert call_kwargs["aspect_ratio"] == "16:9"
        assert call_kwargs["temperature"] == 0.8

    @pytest.mark.asyncio
    @patch("pixelforge_mcp.utils.api_client.GeminiImageGenerator")
    async def test_generate_failure_no_images(self, mock_generator_class, tmp_path):
        """Test generation failure when no images are returned."""
        mock_generator = Mock()
        mock_result = Mock()
        mock_result.images = None
        mock_generator.generate = AsyncMock(return_value=mock_result)
        mock_generator_class.return_value = mock_generator

        client = ImagenAPIClient()
        output_path = tmp_path / "test.png"

        result = await client.generate(prompt="test", output_path=output_path)

        assert result.success is False
        assert result.error == "No images generated"

    @pytest.mark.asyncio
    @patch("pixelforge_mcp.utils.api_client.GeminiImageGenerator")
    async def test_generate_exception(self, mock_generator_class, tmp_path):
        """Test generation with exception."""
        mock_generator = Mock()
        mock_generator.generate = AsyncMock(side_effect=Exception("API error"))
        mock_generator_class.return_value = mock_generator

        client = ImagenAPIClient()
        output_path = tmp_path / "test.png"

        result = await client.generate(prompt="test", output_path=output_path)

        assert result.success is False
        assert "API error" in result.error

    @pytest.mark.asyncio
    @patch("pixelforge_mcp.utils.api_client.GeminiImageGenerator")
    async def test_edit_success(self, mock_generator_class, tmp_path):
        """Test successful image editing."""
        mock_generator = Mock()
        mock_result = Mock()
        mock_result.images = [Mock(spec=Image.Image)]
        mock_generator.generate = AsyncMock(return_value=mock_result)
        mock_generator_class.return_value = mock_generator

        client = ImagenAPIClient()
        input_path = tmp_path / "input.png"
        input_path.touch()
        output_path = tmp_path / "output.png"

        result = await client.edit(
            prompt="add clouds",
            input_path=input_path,
            output_path=output_path,
            temperature=0.7,
        )

        assert result.success is True
        assert result.images == mock_result.images
        assert result.image_paths == [str(output_path)]

        # Verify generate was called with input image
        call_kwargs = mock_generator.generate.call_args[1]
        assert call_kwargs["prompt"] == "add clouds"
        assert call_kwargs["input_images"] == [str(input_path)]
        assert call_kwargs["output_images"] == [str(output_path)]
        assert call_kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    @patch("pixelforge_mcp.utils.api_client.GeminiImageGenerator")
    async def test_edit_failure(self, mock_generator_class, tmp_path):
        """Test failed image editing."""
        mock_generator = Mock()
        mock_result = Mock()
        mock_result.images = None
        mock_generator.generate = AsyncMock(return_value=mock_result)
        mock_generator_class.return_value = mock_generator

        client = ImagenAPIClient()
        input_path = tmp_path / "input.png"
        input_path.touch()
        output_path = tmp_path / "output.png"

        result = await client.edit(
            prompt="add clouds", input_path=input_path, output_path=output_path
        )

        assert result.success is False
        assert result.error == "No images generated"

    @pytest.mark.asyncio
    @patch("pixelforge_mcp.utils.api_client.GeminiImageGenerator")
    async def test_analyze_success(self, mock_generator_class, tmp_path):
        """Test successful image analysis."""
        mock_generator = Mock()
        mock_result = Mock()
        mock_result.text = "A beautiful sunset over mountains"
        mock_generator.generate = AsyncMock(return_value=mock_result)
        mock_generator_class.return_value = mock_generator

        client = ImagenAPIClient()
        image_path = tmp_path / "test.png"
        image_path.touch()

        result = await client.analyze(image_path)

        assert result.success is True
        assert result.output == "A beautiful sunset over mountains"
        assert result.data["analysis"] == "A beautiful sunset over mountains"
        assert str(image_path) in result.data["image_path"]

        # Verify generate was called with default prompt and output_text=True
        call_kwargs = mock_generator.generate.call_args[1]
        assert call_kwargs["prompt"] == DEFAULT_ANALYSIS_PROMPT
        assert call_kwargs["input_images"] == [str(image_path)]
        assert call_kwargs["output_text"] is True

    @pytest.mark.asyncio
    @patch("pixelforge_mcp.utils.api_client.GeminiImageGenerator")
    async def test_analyze_with_custom_prompt(self, mock_generator_class, tmp_path):
        """Test image analysis with a custom prompt."""
        mock_generator = Mock()
        mock_result = Mock()
        mock_result.text = "Text found: Hello World"
        mock_generator.generate = AsyncMock(return_value=mock_result)
        mock_generator_class.return_value = mock_generator

        client = ImagenAPIClient()
        image_path = tmp_path / "test.png"
        image_path.touch()

        custom_prompt = "Extract all visible text from this image"
        result = await client.analyze(image_path, prompt=custom_prompt)

        assert result.success is True
        assert result.output == "Text found: Hello World"

        # Verify custom prompt was used instead of default
        call_kwargs = mock_generator.generate.call_args[1]
        assert call_kwargs["prompt"] == custom_prompt
        assert call_kwargs["input_images"] == [str(image_path)]
        assert call_kwargs["output_text"] is True

    @pytest.mark.asyncio
    @patch("pixelforge_mcp.utils.api_client.GeminiImageGenerator")
    async def test_analyze_failure(self, mock_generator_class, tmp_path):
        """Test failed image analysis."""
        mock_generator = Mock()
        mock_result = Mock()
        mock_result.text = None
        mock_generator.generate = AsyncMock(return_value=mock_result)
        mock_generator_class.return_value = mock_generator

        client = ImagenAPIClient()
        image_path = tmp_path / "test.png"
        image_path.touch()

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
        assert len(models) == 3

        model_names = [m["name"] for m in models]
        assert "gemini-2.5-flash-image" in model_names
        assert "gemini-3-pro-image-preview" in model_names
        assert "gemini-3.1-flash-image-preview" in model_names

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
    @patch("pixelforge_mcp.utils.api_client.GeminiImageGenerator")
    async def test_generator_caching(self, mock_generator_class):
        """Test that generator instance is cached."""
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        client = ImagenAPIClient()

        generator1 = client._get_generator()
        assert generator1 == mock_generator
        assert mock_generator_class.call_count == 1

        generator2 = client._get_generator()
        assert generator2 == mock_generator
        assert mock_generator_class.call_count == 1

    @pytest.mark.asyncio
    @patch("pixelforge_mcp.utils.api_client.GeminiImageGenerator")
    async def test_model_override(self, mock_generator_class, tmp_path):
        """Test model override in generate call."""
        mock_generator = Mock()
        mock_result = Mock()
        mock_result.images = [Mock(spec=Image.Image)]
        mock_generator.generate = AsyncMock(return_value=mock_result)
        mock_generator.model_name = "gemini-2.5-flash-image"
        mock_generator_class.return_value = mock_generator

        client = ImagenAPIClient(model_name="gemini-2.5-flash-image")
        output_path = tmp_path / "test.png"

        await client.generate(
            prompt="test",
            output_path=output_path,
            model="gemini-3-pro-image-preview",
        )

        assert mock_generator.model_name == "gemini-3-pro-image-preview"


class TestSDKRouting:
    """Tests for direct SDK routing logic."""

    def test_needs_direct_sdk_with_image_size(self):
        """Test routing to SDK when image_size is set."""
        assert ImagenAPIClient._needs_direct_sdk(image_size="4K") is True

    def test_needs_direct_sdk_with_person_generation(self):
        """Test routing to SDK when person_generation is set."""
        assert ImagenAPIClient._needs_direct_sdk(person_generation="allow") is True

    def test_needs_direct_sdk_with_both(self):
        """Test routing to SDK when both extended params are set."""
        assert (
            ImagenAPIClient._needs_direct_sdk(
                image_size="2K", person_generation="block"
            )
            is True
        )

    def test_no_sdk_needed_for_basic_call(self):
        """Test wrapper path used for basic calls."""
        assert ImagenAPIClient._needs_direct_sdk() is False
        assert ImagenAPIClient._needs_direct_sdk(image_size=None) is False
        assert (
            ImagenAPIClient._needs_direct_sdk(image_size=None, person_generation=None)
            is False
        )


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
    @patch("pixelforge_mcp.utils.api_client.GeminiImageGenerator")
    async def test_edit_with_model_override(self, mock_generator_class, tmp_path):
        """Test edit applies model override."""
        mock_generator = Mock()
        mock_result = Mock()
        mock_result.images = [Mock(spec=Image.Image)]
        mock_generator.generate = AsyncMock(return_value=mock_result)
        mock_generator.model_name = "gemini-2.5-flash-image"
        mock_generator_class.return_value = mock_generator

        client = ImagenAPIClient()
        input_path = tmp_path / "input.png"
        input_path.touch()
        output_path = tmp_path / "output.png"

        result = await client.edit(
            prompt="add clouds",
            input_path=input_path,
            output_path=output_path,
            model="gemini-3-pro-image-preview",
        )

        assert result.success is True
        assert mock_generator.model_name == "gemini-3-pro-image-preview"

    @pytest.mark.asyncio
    @patch("pixelforge_mcp.utils.api_client.GeminiImageGenerator")
    async def test_edit_without_model_uses_default(
        self, mock_generator_class, tmp_path
    ):
        """Test edit without model uses the client default."""
        mock_generator = Mock()
        mock_result = Mock()
        mock_result.images = [Mock(spec=Image.Image)]
        mock_generator.generate = AsyncMock(return_value=mock_result)
        mock_generator_class.return_value = mock_generator

        client = ImagenAPIClient(model_name="gemini-2.5-flash-image")
        input_path = tmp_path / "input.png"
        input_path.touch()
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


class TestSDKRoutingExtended:
    """Tests for extended SDK routing with reference_images."""

    def test_needs_sdk_with_reference_images(self):
        """Test routing to SDK when reference_images are set."""
        assert (
            ImagenAPIClient._needs_direct_sdk(reference_images=["/path/to/ref.png"])
            is True
        )

    def test_needs_sdk_with_all_extended(self):
        """Test routing when all extended params are set."""
        assert (
            ImagenAPIClient._needs_direct_sdk(
                image_size="4K",
                person_generation="allow",
                reference_images=["/path/to/ref.png"],
            )
            is True
        )

    def test_no_sdk_without_extended(self):
        """Test wrapper path for basic calls."""
        assert (
            ImagenAPIClient._needs_direct_sdk(
                image_size=None,
                person_generation=None,
                reference_images=None,
            )
            is False
        )
