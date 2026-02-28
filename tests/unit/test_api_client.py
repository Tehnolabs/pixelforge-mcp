"""Unit tests for API client using gemini-imagen library."""

import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from PIL import Image

from pixelforge_mcp.utils.api_client import (
    DEFAULT_ANALYSIS_PROMPT,
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
            model_name="gemini-3-pro-image-preview", api_key="test-key", log_images=True
        )
        assert client.model_name == "gemini-3-pro-image-preview"
        assert client.api_key == "test-key"
        assert client.log_images is True

    @pytest.mark.asyncio
    @patch("pixelforge_mcp.utils.api_client.GeminiImageGenerator")
    async def test_generate_success(self, mock_generator_class, tmp_path):
        """Test successful image generation."""
        # Mock the generator instance and its generate method
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

        # Check models is a list of dicts with metadata
        models = result.data["models"]
        assert isinstance(models, list)
        assert len(models) == 3

        # Verify model structure
        model_names = [m["name"] for m in models]
        assert "gemini-2.5-flash-image" in model_names
        assert "gemini-3-pro-image-preview" in model_names
        assert "gemini-3.1-flash-image-preview" in model_names

        # Verify metadata fields exist
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
    @patch("pixelforge_mcp.utils.api_client.GeminiImageGenerator")
    async def test_generator_caching(self, mock_generator_class):
        """Test that generator instance is cached."""
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        client = ImagenAPIClient()

        # First call creates generator
        generator1 = client._get_generator()
        assert generator1 == mock_generator
        assert mock_generator_class.call_count == 1

        # Second call returns cached instance
        generator2 = client._get_generator()
        assert generator2 == mock_generator
        assert mock_generator_class.call_count == 1  # Not called again

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
            prompt="test", output_path=output_path, model="gemini-3-pro-image-preview"
        )

        # Verify model was changed
        assert mock_generator.model_name == "gemini-3-pro-image-preview"
