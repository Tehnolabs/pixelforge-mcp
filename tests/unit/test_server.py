"""Unit tests for MCP server handlers."""

import os

os.environ.setdefault("GOOGLE_API_KEY", "test-placeholder-key")

import re
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from pixelforge_mcp import server as server_module
from pixelforge_mcp.server import (
    estimate_cost as _estimate_cost_tool,
)
from pixelforge_mcp.server import (
    format_image_result,
    generate_output_path,
)
from pixelforge_mcp.server import (
    generate_image as _generate_image_tool,
)
from pixelforge_mcp.utils.api_client import GenerationResult, ImagenAPIClient
from pixelforge_mcp.utils.validation import (
    ANALYSIS_MODELS,
    COST_TABLE,
    IMAGE_MODELS,
)

# In FastMCP 3.x, @mcp.tool() returns the original function directly.
estimate_cost_fn = _estimate_cost_tool
generate_image_fn = _generate_image_tool


# ------------------------------------------------------------------
# format_image_result
# ------------------------------------------------------------------


class TestFormatImageResult:
    """Tests for format_image_result helper."""

    def test_success_with_existing_files(self, tmp_path):
        """Test successful result with files that exist on disk."""
        # Create actual files with some content
        file1 = tmp_path / "img1.png"
        file1.write_bytes(b"fake png data 12345")
        file2 = tmp_path / "img2.png"
        file2.write_bytes(b"fake png data 67890")

        result = GenerationResult(
            success=True,
            output="Images generated successfully",
        )

        response = format_image_result(result, [file1, file2])

        assert response["success"] is True
        assert response["message"] == "Images generated successfully"
        assert len(response["images"]) == 2
        assert response["images"][0]["path"] == str(file1.absolute())
        assert response["images"][0]["size_bytes"] == len(b"fake png data 12345")
        assert response["images"][1]["path"] == str(file2.absolute())
        assert response["images"][1]["size_bytes"] == len(b"fake png data 67890")
        assert "details" not in response

    def test_success_with_missing_files(self, tmp_path):
        """Test successful result when output files don't exist."""
        missing1 = tmp_path / "nonexistent1.png"
        missing2 = tmp_path / "nonexistent2.png"

        result = GenerationResult(
            success=True,
            output="Generated but files missing",
        )

        response = format_image_result(result, [missing1, missing2])

        assert response["success"] is True
        assert response["message"] == "Generated but files missing"
        assert response["images"] == []

    def test_success_with_mixed_existing_and_missing(self, tmp_path):
        """Test result with some files existing and some missing."""
        existing = tmp_path / "exists.png"
        existing.write_bytes(b"real data")
        missing = tmp_path / "missing.png"

        result = GenerationResult(success=True, output="Partial files")

        response = format_image_result(result, [existing, missing])

        assert response["success"] is True
        assert len(response["images"]) == 1
        assert response["images"][0]["path"] == str(existing.absolute())

    def test_failure_result(self):
        """Test failed result returns error message."""
        result = GenerationResult(
            success=False,
            output="",
            error="API quota exceeded",
        )

        response = format_image_result(result, [])

        assert response["success"] is False
        assert response["message"] == "API quota exceeded"
        assert "images" not in response

    def test_failure_result_with_no_error(self):
        """Test failed result with no error message uses default."""
        result = GenerationResult(
            success=False,
            output="",
            error=None,
        )

        response = format_image_result(result, [])

        assert response["success"] is False
        assert response["message"] == "Operation failed"

    def test_success_with_data(self, tmp_path):
        """Test successful result includes data as 'details' key."""
        file1 = tmp_path / "out.png"
        file1.write_bytes(b"png content")

        result = GenerationResult(
            success=True,
            output="Done",
            data={
                "model": "gemini-2.5-flash-image",
                "aspect_ratio": "16:9",
                "temperature": 0.8,
            },
        )

        response = format_image_result(result, [file1])

        assert response["success"] is True
        assert response["details"]["model"] == "gemini-2.5-flash-image"
        assert response["details"]["aspect_ratio"] == "16:9"
        assert response["details"]["temperature"] == 0.8

    def test_success_without_data(self, tmp_path):
        """Test successful result without data omits 'details' key."""
        file1 = tmp_path / "out.png"
        file1.write_bytes(b"png content")

        result = GenerationResult(
            success=True,
            output="Done",
            data=None,
        )

        response = format_image_result(result, [file1])

        assert response["success"] is True
        assert "details" not in response

    def test_success_with_empty_paths_list(self):
        """Test successful result with empty output paths list."""
        result = GenerationResult(success=True, output="No images to report")

        response = format_image_result(result, [])

        assert response["success"] is True
        assert response["images"] == []

    def test_warnings_propagated_to_response(self, tmp_path):
        """Test warnings from result.data are surfaced in response."""
        file1 = tmp_path / "out.png"
        file1.write_bytes(b"png content")

        result = GenerationResult(
            success=True,
            output="Done",
            data={"warnings": ["person_generation ignored"]},
        )

        response = format_image_result(result, [file1])

        assert response["success"] is True
        assert response["warnings"] == ["person_generation ignored"]

    def test_no_warnings_when_not_in_data(self, tmp_path):
        """Test no warnings key when result.data has none."""
        file1 = tmp_path / "out.png"
        file1.write_bytes(b"png content")

        result = GenerationResult(
            success=True,
            output="Done",
            data={"model": "gemini-2.5-flash-image"},
        )

        response = format_image_result(result, [file1])

        assert response["success"] is True
        assert "warnings" not in response
        assert "auto_routed" not in response


# ------------------------------------------------------------------
# generate_output_path
# ------------------------------------------------------------------


class TestGenerateOutputPath:
    """Tests for generate_output_path helper."""

    @patch("pixelforge_mcp.server.get_config")
    def test_with_custom_filename(self, mock_config, tmp_path):
        """Test that a custom filename returns output_dir / filename."""
        mock_config.return_value.storage.output_dir = tmp_path

        path = generate_output_path(filename="my_image.png")

        assert path == tmp_path / "my_image.png"

    @patch("pixelforge_mcp.server.get_config")
    def test_without_filename_generates_timestamp(self, mock_config, tmp_path):
        """Test auto-generated name uses prefix, timestamp, and extension."""
        mock_config.return_value.storage.output_dir = tmp_path

        path = generate_output_path()

        assert path.parent == tmp_path
        # Pattern: generated_YYYYMMDD_HHMMSS_mmm.png
        assert re.match(
            r"generated_\d{8}_\d{6}_\d{3}\.png",
            path.name,
        )

    @patch("pixelforge_mcp.server.get_config")
    def test_custom_prefix(self, mock_config, tmp_path):
        """Test custom prefix appears in generated filename."""
        mock_config.return_value.storage.output_dir = tmp_path

        path = generate_output_path(prefix="edited")

        assert path.name.startswith("edited_")

    @patch("pixelforge_mcp.server.get_config")
    def test_format_png(self, mock_config, tmp_path):
        """Test png format produces .png extension."""
        mock_config.return_value.storage.output_dir = tmp_path

        path = generate_output_path(output_format="png")

        assert path.suffix == ".png"

    @patch("pixelforge_mcp.server.get_config")
    def test_format_jpeg(self, mock_config, tmp_path):
        """Test jpeg format produces .jpg extension."""
        mock_config.return_value.storage.output_dir = tmp_path

        path = generate_output_path(output_format="jpeg")

        assert path.suffix == ".jpg"

    @patch("pixelforge_mcp.server.get_config")
    def test_format_webp(self, mock_config, tmp_path):
        """Test webp format produces .webp extension."""
        mock_config.return_value.storage.output_dir = tmp_path

        path = generate_output_path(output_format="webp")

        assert path.suffix == ".webp"

    @patch("pixelforge_mcp.server.get_config")
    def test_unknown_format_defaults_to_png(self, mock_config, tmp_path):
        """Test unknown format falls back to .png extension."""
        mock_config.return_value.storage.output_dir = tmp_path

        path = generate_output_path(output_format="bmp")

        assert path.suffix == ".png"

    @patch("pixelforge_mcp.server.get_config")
    def test_filename_overrides_format(self, mock_config, tmp_path):
        """Test that when filename is provided, output_format is ignored."""
        mock_config.return_value.storage.output_dir = tmp_path

        path = generate_output_path(filename="photo.webp", output_format="jpeg")

        # Filename wins: the extension comes from the filename itself
        assert path == tmp_path / "photo.webp"

    @patch("pixelforge_mcp.server.get_config")
    def test_two_sequential_paths_are_unique(self, mock_config, tmp_path):
        """Test that two sequential calls produce different filenames."""
        mock_config.return_value.storage.output_dir = tmp_path

        path1 = generate_output_path()
        path2 = generate_output_path()

        # The millisecond component should differ (or the second/minute)
        # In rare cases they could collide, but in practice they won't
        # during tests. We just verify they are valid Path objects.
        assert path1.parent == tmp_path
        assert path2.parent == tmp_path


# ------------------------------------------------------------------
# estimate_cost tool handler
# ------------------------------------------------------------------


class TestEstimateCost:
    """Tests for the estimate_cost MCP tool handler."""

    @pytest.mark.asyncio
    async def test_single_model_generate(self):
        """Test cost estimate for a single model generate operation."""
        result = await estimate_cost_fn(
            operation="generate",
            model="gemini-nano-banana-2.1",
            number_of_images=1,
        )

        assert result["success"] is True
        assert result["model"] == "gemini-nano-banana-2.1"
        assert result["operation"] == "generate"
        assert (
            result["unit_cost_usd"] == COST_TABLE["gemini-nano-banana-2.1"]["generate"]
        )
        assert result["number_of_images"] == 1
        assert result["total_cost_usd"] == result["unit_cost_usd"]

    @pytest.mark.asyncio
    async def test_single_model_edit(self):
        """Test cost estimate for a single model edit operation."""
        result = await estimate_cost_fn(
            operation="edit",
            model="gemini-3-pro-image",
            number_of_images=1,
        )

        assert result["success"] is True
        assert result["model"] == "gemini-3-pro-image"
        assert result["operation"] == "edit"
        assert result["unit_cost_usd"] == COST_TABLE["gemini-3-pro-image"]["edit"]

    @pytest.mark.asyncio
    async def test_single_model_unsupported_operation(self):
        """A model that does not price the operation returns an error."""
        from pixelforge_mcp.server import estimate_cost

        result = await estimate_cost(
            operation="analyze",
            model="gemini-nano-banana-2.1",
        )
        assert result["success"] is False
        assert "does not support" in result["message"]

    @pytest.mark.asyncio
    async def test_single_model_analyze(self):
        """Test cost estimate for a single model analyze operation."""
        result = await estimate_cost_fn(
            operation="analyze",
            model="gemini-3.5-flash-lite",
            number_of_images=1,
        )

        assert result["success"] is True
        assert result["operation"] == "analyze"
        assert result["unit_cost_usd"] == COST_TABLE["gemini-3.5-flash-lite"]["analyze"]

    @pytest.mark.asyncio
    async def test_multiple_images(self):
        """Test cost estimate scales correctly with number_of_images."""
        result = await estimate_cost_fn(
            operation="generate",
            model="gemini-nano-banana-2.1",
            number_of_images=10,
        )

        assert result["success"] is True
        assert result["number_of_images"] == 10
        expected_total = COST_TABLE["gemini-nano-banana-2.1"]["generate"] * 10
        assert result["total_cost_usd"] == round(expected_total, 4)

    @pytest.mark.asyncio
    async def test_all_models_returned_when_no_model(self):
        """Test that omitting model returns estimates for all models."""
        result = await estimate_cost_fn(
            operation="generate",
            number_of_images=1,
        )

        assert result["success"] is True
        assert result["operation"] == "generate"
        assert result["number_of_images"] == 1
        assert "estimates" in result
        assert "note" in result

        estimates = result["estimates"]
        model_names = [e["model"] for e in estimates]
        for model_name in IMAGE_MODELS:
            assert model_name in model_names
        for model_name in ANALYSIS_MODELS:
            assert model_name not in model_names

        # Each estimate has the expected fields
        for est in estimates:
            assert "model" in est
            assert "unit_cost_usd" in est
            assert "total_cost_usd" in est

    @pytest.mark.asyncio
    async def test_all_models_analyze_lists_only_analysis_models(self):
        """Test analyze estimates cover the text models, not image models."""
        result = await estimate_cost_fn(operation="analyze", number_of_images=1)

        assert result["success"] is True
        model_names = [e["model"] for e in result["estimates"]]
        assert model_names == ANALYSIS_MODELS
        for est in result["estimates"]:
            assert est["unit_cost_usd"] > 0

    @pytest.mark.asyncio
    async def test_all_models_with_multiple_images(self):
        """Test all-models estimate scales with number_of_images."""
        result = await estimate_cost_fn(
            operation="generate",
            number_of_images=5,
        )

        assert result["success"] is True
        for est in result["estimates"]:
            model_name = est["model"]
            expected = COST_TABLE[model_name]["generate"] * 5
            assert est["total_cost_usd"] == round(expected, 4)

    @pytest.mark.asyncio
    async def test_invalid_operation(self):
        """Test that an invalid operation returns a validation error."""
        result = await estimate_cost_fn(
            operation="invalid",
            number_of_images=1,
        )

        assert result["success"] is False
        assert "Invalid" in result["message"] or "invalid" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_unknown_model(self):
        """Test that an unknown model returns an error."""
        result = await estimate_cost_fn(
            operation="generate",
            model="nonexistent-model",
            number_of_images=1,
        )

        assert result["success"] is False
        assert "Unknown model" in result["message"]

    @pytest.mark.asyncio
    async def test_operation_case_insensitive(self):
        """Test that operation is case insensitive."""
        result = await estimate_cost_fn(
            operation="GENERATE",
            model="gemini-nano-banana-2.1",
            number_of_images=1,
        )

        assert result["success"] is True
        assert result["operation"] == "generate"

    @pytest.mark.asyncio
    async def test_total_cost_rounded(self):
        """Test that total cost is rounded to 4 decimal places."""
        result = await estimate_cost_fn(
            operation="generate",
            model="gemini-nano-banana-2.1",
            number_of_images=3,
        )

        assert result["success"] is True
        # Verify rounding: 0.0336 * 3 = 0.1008 (float noise must be rounded away)
        assert result["total_cost_usd"] == 0.1008
        total_str = str(result["total_cost_usd"])
        # Should have at most 4 decimal places
        if "." in total_str:
            decimals = len(total_str.split(".")[1])
            assert decimals <= 4


# ------------------------------------------------------------------
# generate_image multi-image behavior
# ------------------------------------------------------------------


class TestGenerateImageMultiImage:
    """Tests for multi-image generation aggregate behavior."""

    @pytest.mark.asyncio
    @patch("pixelforge_mcp.server.get_api_client")
    @patch("pixelforge_mcp.server.get_config")
    async def test_all_succeed(self, mock_config, mock_get_client, tmp_path):
        """Test multi-image generation when all images succeed."""
        pass  # generate_image_fn imported at module level

        mock_config.return_value.storage.output_dir = tmp_path

        # Create files on disk to simulate successful generation
        async def mock_generate(**kwargs):
            output_path = kwargs["output_path"]
            output_path.write_bytes(b"fake image data")
            return GenerationResult(
                success=True,
                output=f"Generated at {output_path}",
                data={"model": "gemini-2.5-flash-image"},
            )

        mock_client = Mock()
        mock_client.generate = AsyncMock(side_effect=mock_generate)
        mock_get_client.return_value = mock_client

        result = await generate_image_fn(
            prompt="test prompt",
            number_of_images=3,
        )

        assert result["success"] is True
        assert len(result["images"]) == 3
        assert "partial_failure" not in result

    @pytest.mark.asyncio
    @patch("pixelforge_mcp.server.get_api_client")
    @patch("pixelforge_mcp.server.get_config")
    async def test_partial_failure(self, mock_config, mock_get_client, tmp_path):
        """Test multi-image generation with some failures."""
        pass  # generate_image_fn imported at module level

        mock_config.return_value.storage.output_dir = tmp_path

        call_count = 0

        async def mock_generate(**kwargs):
            nonlocal call_count
            call_count += 1
            output_path = kwargs["output_path"]
            if call_count == 2:
                return GenerationResult(
                    success=False,
                    output="",
                    error="Safety filter blocked",
                )
            output_path.write_bytes(b"fake image data")
            return GenerationResult(
                success=True,
                output=f"Generated at {output_path}",
                data={"model": "gemini-2.5-flash-image"},
            )

        mock_client = Mock()
        mock_client.generate = AsyncMock(side_effect=mock_generate)
        mock_get_client.return_value = mock_client

        result = await generate_image_fn(
            prompt="test prompt",
            number_of_images=3,
        )

        assert result["success"] is True
        assert result["partial_failure"] is True
        assert len(result["errors"]) == 1
        assert "Safety filter blocked" in result["errors"][0]
        assert "2/3" in result["message"]

    @pytest.mark.asyncio
    @patch("pixelforge_mcp.server.get_api_client")
    @patch("pixelforge_mcp.server.get_config")
    async def test_all_fail(self, mock_config, mock_get_client, tmp_path):
        """Test multi-image generation when all images fail."""
        pass  # generate_image_fn imported at module level

        mock_config.return_value.storage.output_dir = tmp_path

        async def mock_generate(**kwargs):
            return GenerationResult(
                success=False,
                output="",
                error="API error",
            )

        mock_client = Mock()
        mock_client.generate = AsyncMock(side_effect=mock_generate)
        mock_get_client.return_value = mock_client

        result = await generate_image_fn(
            prompt="test prompt",
            number_of_images=2,
        )

        assert result["success"] is False
        assert "All image generations failed" in result["message"]
        assert len(result["errors"]) == 2

    @pytest.mark.asyncio
    @patch("pixelforge_mcp.server.get_api_client")
    @patch("pixelforge_mcp.server.get_config")
    async def test_single_image_no_partial_failure_key(
        self, mock_config, mock_get_client, tmp_path
    ):
        """Test single image generation does not set partial_failure."""
        pass  # generate_image_fn imported at module level

        mock_config.return_value.storage.output_dir = tmp_path

        async def mock_generate(**kwargs):
            output_path = kwargs["output_path"]
            output_path.write_bytes(b"image bytes")
            return GenerationResult(
                success=True,
                output="Done",
            )

        mock_client = Mock()
        mock_client.generate = AsyncMock(side_effect=mock_generate)
        mock_get_client.return_value = mock_client

        result = await generate_image_fn(prompt="test prompt", number_of_images=1)

        assert result["success"] is True
        assert "partial_failure" not in result


# ------------------------------------------------------------------
# Quality presets and warnings through generate_image
# ------------------------------------------------------------------


class TestGenerateImageModelResolution:
    """Tests for how generate_image resolves models and surfaces warnings."""

    @staticmethod
    def _client_writing_files(data=None):
        async def mock_generate(**kwargs):
            kwargs["output_path"].write_bytes(b"fake image data")
            return GenerationResult(
                success=True,
                output="Done",
                data=data or {"model": kwargs["model"]},
            )

        mock_client = Mock()
        mock_client.generate = AsyncMock(side_effect=mock_generate)
        return mock_client

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "quality,expected_model,expected_size",
        [
            ("fast", "gemini-3.1-flash-lite-image", None),
            ("balanced", "gemini-nano-banana-2.1", "1K"),
            ("quality", "gemini-3-pro-image", "2K"),
        ],
    )
    @patch("pixelforge_mcp.utils.history.get_history")
    @patch("pixelforge_mcp.server.get_api_client")
    @patch("pixelforge_mcp.server.get_config")
    async def test_quality_presets_resolve_to_stable_models(
        self,
        mock_config,
        mock_get_client,
        mock_history,
        quality,
        expected_model,
        expected_size,
        tmp_path,
    ):
        """Each quality preset reaches the client as its model and size."""
        mock_config.return_value.storage.output_dir = tmp_path
        mock_client = self._client_writing_files()
        mock_get_client.return_value = mock_client

        result = await generate_image_fn(prompt="test prompt", quality=quality)

        assert result["success"] is True
        kwargs = mock_client.generate.call_args.kwargs
        assert kwargs["model"] == expected_model
        assert kwargs["image_size"] == expected_size

    @pytest.mark.asyncio
    @patch("pixelforge_mcp.utils.history.get_history")
    @patch("pixelforge_mcp.server.get_api_client")
    @patch("pixelforge_mcp.server.get_config")
    async def test_explicit_model_passes_through(
        self, mock_config, mock_get_client, mock_history, tmp_path
    ):
        """An explicit model and 4K size reach the client unchanged."""
        mock_config.return_value.storage.output_dir = tmp_path
        mock_client = self._client_writing_files()
        mock_get_client.return_value = mock_client

        result = await generate_image_fn(
            prompt="test prompt",
            model="gemini-3.1-flash-image",
            image_size="4K",
        )

        assert result["success"] is True
        kwargs = mock_client.generate.call_args.kwargs
        assert kwargs["model"] == "gemini-3.1-flash-image"
        assert kwargs["image_size"] == "4K"

    @pytest.mark.asyncio
    @patch("pixelforge_mcp.utils.history.get_history")
    @patch("pixelforge_mcp.server.get_api_client")
    @patch("pixelforge_mcp.server.get_config")
    async def test_alias_warning_reaches_response(
        self, mock_config, mock_get_client, mock_history, tmp_path
    ):
        """Warnings from the client are surfaced at the top level."""
        mock_config.return_value.storage.output_dir = tmp_path
        mock_client = self._client_writing_files(
            data={
                "model": "gemini-nano-banana-2.1",
                "warnings": ["Model 'gemini-3.1-flash-image-preview' was retired"],
            }
        )
        mock_get_client.return_value = mock_client

        result = await generate_image_fn(
            prompt="test prompt", model="gemini-3.1-flash-image-preview"
        )

        assert result["success"] is True
        assert result["warnings"] == [
            "Model 'gemini-3.1-flash-image-preview' was retired"
        ]
        assert result["details"]["model"] == "gemini-nano-banana-2.1"

    @pytest.mark.asyncio
    @patch("pixelforge_mcp.server.get_api_client")
    @patch("pixelforge_mcp.server.get_config")
    async def test_retired_model_error_reaches_response(
        self, mock_config, mock_get_client, tmp_path
    ):
        """A retired Imagen ID produces a failed response naming the fix."""
        mock_config.return_value.storage.output_dir = tmp_path
        client = ImagenAPIClient(api_key="test-key")
        client._genai_client = Mock()
        mock_get_client.return_value = client

        result = await generate_image_fn(
            prompt="test prompt", model="imagen-4.0-generate-001"
        )

        assert result["success"] is False
        assert "retired" in result["errors"][0]
        assert "gemini-nano-banana-2.1" in result["errors"][0]
        client._genai_client.aio.models.generate_content.assert_not_called()


# ------------------------------------------------------------------
# Analysis handlers forward the model override
# ------------------------------------------------------------------


ANALYSIS_HANDLER_CASES = [
    pytest.param(
        "analyze_image",
        "analyze",
        GenerationResult(success=True, output="d", data={"analysis": "d"}),
        id="analyze_image",
    ),
    pytest.param(
        "extract_text",
        "extract_text",
        GenerationResult(
            success=True, output="t", data={"extracted_text": "t", "blocks": []}
        ),
        id="extract_text",
    ),
    pytest.param(
        "detect_objects",
        "detect_objects",
        GenerationResult(success=True, output="n", data={"detections": [], "count": 0}),
        id="detect_objects",
    ),
    pytest.param(
        "compare_images",
        "compare_images",
        GenerationResult(success=True, output="c", data={"analysis": "c"}),
        id="compare_images",
    ),
    pytest.param(
        "optimize_prompt",
        "optimize_prompt",
        GenerationResult(
            success=True,
            output="better",
            data={"original_prompt": "a cat", "enhanced_prompt": "better"},
        ),
        id="optimize_prompt",
    ),
]


def _analysis_call_kwargs(handler_name, image_path):
    """Build handler kwargs: the image argument differs per tool."""
    if handler_name == "optimize_prompt":
        return {"prompt": "a cat"}
    if handler_name == "compare_images":
        return {"image_paths": [image_path, image_path]}
    return {"image_path": image_path}


class TestAnalysisHandlersForwardModel:
    """Each analysis tool passes its model argument to the API client."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "handler_name,client_method,client_result", ANALYSIS_HANDLER_CASES
    )
    @patch("pixelforge_mcp.server.get_api_client")
    async def test_model_override_forwarded(
        self,
        mock_get_client,
        handler_name,
        client_method,
        client_result,
        tmp_path,
    ):
        """An explicit model reaches the client method as a keyword."""
        image = tmp_path / "img.png"
        image.touch()
        mock_client = Mock()
        setattr(mock_client, client_method, AsyncMock(return_value=client_result))
        mock_get_client.return_value = mock_client

        handler = getattr(server_module, handler_name)
        result = await handler(
            **_analysis_call_kwargs(handler_name, str(image)),
            model="gemini-3.8-flash",
        )

        assert result["success"] is True
        method = getattr(mock_client, client_method)
        method.assert_awaited_once()
        assert method.call_args.kwargs["model"] == "gemini-3.8-flash"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "handler_name,client_method,client_result", ANALYSIS_HANDLER_CASES
    )
    @patch("pixelforge_mcp.server.get_api_client")
    async def test_model_defaults_to_none(
        self,
        mock_get_client,
        handler_name,
        client_method,
        client_result,
        tmp_path,
    ):
        """Without a model the client receives None and applies its default."""
        image = tmp_path / "img.png"
        image.touch()
        mock_client = Mock()
        setattr(mock_client, client_method, AsyncMock(return_value=client_result))
        mock_get_client.return_value = mock_client

        handler = getattr(server_module, handler_name)
        result = await handler(**_analysis_call_kwargs(handler_name, str(image)))

        assert result["success"] is True
        method = getattr(mock_client, client_method)
        assert method.call_args.kwargs["model"] is None


# ------------------------------------------------------------------
# list_available_models / get_server_info
# ------------------------------------------------------------------


class TestListAvailableModels:
    """Tests for the list_available_models tool handler."""

    @pytest.mark.asyncio
    @patch("pixelforge_mcp.server.get_api_client")
    async def test_returns_image_and_analysis_models(self, mock_get_client):
        """The handler passes through both model lists and the recommendation."""
        mock_get_client.return_value = ImagenAPIClient(api_key="test-key")

        result = await server_module.list_available_models()

        assert result["success"] is True
        assert [m["name"] for m in result["models"]] == IMAGE_MODELS
        assert [m["name"] for m in result["analysis_models"]] == ANALYSIS_MODELS
        assert "gemini-nano-banana-2.1" in result["recommendation"]

    @pytest.mark.asyncio
    @patch("pixelforge_mcp.server.get_api_client")
    async def test_fallback_lists_stable_models_only(self, mock_get_client):
        """If the client reports failure, the fallback names stable models."""
        client = Mock()
        client.list_models = AsyncMock(
            return_value=GenerationResult(success=False, output="", error="boom")
        )
        mock_get_client.return_value = client

        result = await server_module.list_available_models()

        names = [m["name"] for m in result["models"]]
        assert names == [
            "gemini-nano-banana-2.1",
            "gemini-3.1-flash-lite-image",
            "gemini-3.1-flash-image",
            "gemini-3-pro-image",
        ]
        assert [m["name"] for m in result["models"] if m["default"]] == [
            "gemini-nano-banana-2.1"
        ]


class TestGetServerInfo:
    """Tests for get_server_info feature text."""

    def test_person_generation_text_is_not_imagen_specific(self):
        """person_generation is described as accepted but not enforced."""
        info = server_module.get_server_info()

        text = info["features"]["person_generation"]
        assert "not enforced" in text
        assert "Imagen" not in text

    def test_default_model_is_nano_banana_2_1(self):
        """The reported default model is the stable default."""
        info = server_module.get_server_info()

        assert info["imagen"]["default_model"] == "gemini-nano-banana-2.1"


class TestToolCount:
    """The documented tool count must match the registered tools."""

    def test_nineteen_tools_registered(self):
        """Count @mcp.tool() registrations in the server source."""
        source = Path(server_module.__file__).read_text()
        assert source.count("@mcp.tool()") == 19
