"""Integration tests: one live call per MCP tool.

The backend follows the environment:

- Gemini API: ``GOOGLE_API_KEY`` (or ``GEMINI_API_KEY``,
  ``GOOGLE_GENERATIVE_AI_API_KEY``)
- Vertex AI: ``GOOGLE_CLOUD_PROJECT`` plus Application Default Credentials,
  with ``GOOGLE_CLOUD_LOCATION=global``

Without credentials every test is skipped. Each tool is called once with
the cheapest options and one generated image is shared across the tests.

Run with: pytest tests/integration -q -m integration
"""

import asyncio
import os
from pathlib import Path

import pytest
import pytest_asyncio
from PIL import Image

from pixelforge_mcp.config import get_config
from pixelforge_mcp.server import (
    advanced_edit,
    analyze_image,
    edit_image,
    extract_text,
    generate_image,
    get_server_info,
    list_available_models,
    upscale_image,
)

API_KEY_VARS = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY")
LITE_MODEL = "gemini-3.1-flash-lite-image"
# Pause between image-model calls to stay under per-minute quotas.
PAUSE_SECONDS = 3


def _has_credentials() -> bool:
    if any(os.getenv(var) for var in API_KEY_VARS):
        return True
    return bool(os.getenv("GOOGLE_CLOUD_PROJECT"))


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _has_credentials(),
        reason="Set GOOGLE_API_KEY or GOOGLE_CLOUD_PROJECT to run integration tests",
    ),
]


async def _pause() -> None:
    await asyncio.sleep(PAUSE_SECONDS)


@pytest.fixture(scope="module", autouse=True)
def output_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Send generated files to a temporary directory, not ./generated_images.

    output_filename cannot contain path separators, so the output directory
    itself is redirected and each test reads the path the tool returns.
    """
    directory = tmp_path_factory.mktemp("pixelforge")
    get_config().storage.output_dir = directory
    return directory


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def generated_image(output_dir: Path) -> Path:
    """One 1K image from the lite model, reused by every image test."""
    result = await generate_image(
        prompt="A red circle centered on a plain white background, with the "
        "word PIXEL in large black capital letters below it",
        model=LITE_MODEL,
        image_size="1K",
        aspect_ratio="1:1",
    )
    assert result["success"] is True, result.get("message")
    assert len(result["images"]) == 1
    path = Path(result["images"][0]["path"])
    assert path.resolve().parent == output_dir.resolve()
    assert path.is_file()
    assert path.stat().st_size > 0
    with Image.open(path) as img:
        assert max(img.size) == 1024
    return path


def test_get_server_info() -> None:
    result = get_server_info()
    assert result["backend"]["name"] in {"vertex_ai", "gemini_api"}
    assert result["server"]["name"] == "pixelforge-mcp"
    assert "upscale_image" in result["tools"]["generation"]


@pytest.mark.asyncio(loop_scope="module")
async def test_list_available_models() -> None:
    result = await list_available_models()
    assert result["success"] is True
    names = {model["name"] for model in result["models"]}
    assert LITE_MODEL in names


@pytest.mark.asyncio(loop_scope="module")
async def test_generate_image(generated_image: Path) -> None:
    assert generated_image.is_file()


@pytest.mark.asyncio(loop_scope="module")
async def test_analyze_image(generated_image: Path) -> None:
    result = await analyze_image(image_path=str(generated_image))
    assert result["success"] is True, result.get("message")
    assert isinstance(result["analysis"], str)
    assert result["analysis"].strip()


@pytest.mark.asyncio(loop_scope="module")
async def test_extract_text(generated_image: Path) -> None:
    result = await extract_text(image_path=str(generated_image))
    assert result["success"] is True, result.get("message")
    assert isinstance(result["text"], str)
    assert isinstance(result["blocks"], list)


@pytest.mark.asyncio(loop_scope="module")
async def test_edit_image(generated_image: Path) -> None:
    await _pause()
    result = await edit_image(
        prompt="Change the red circle to a blue square",
        input_image_path=str(generated_image),
        model=LITE_MODEL,
    )
    assert result["success"] is True, result.get("message")
    path = Path(result["images"][0]["path"])
    assert path != generated_image
    assert path.is_file()
    assert path.stat().st_size > 0


@pytest.mark.asyncio(loop_scope="module")
async def test_upscale_image(generated_image: Path) -> None:
    await _pause()
    result = await upscale_image(image_path=str(generated_image), upscale_factor="x2")
    assert result["success"] is True, result.get("message")
    assert result["method"] == "generative"
    assert result["image_size"] == "2K"
    assert max(result["source_size"]) == 1024
    assert max(result["output_size"]) == 2048
    path = Path(result["output_path"])
    assert path.is_file()
    with Image.open(path) as img:
        assert max(img.size) == 2048


@pytest.mark.asyncio(loop_scope="module")
async def test_advanced_edit_background_swap(generated_image: Path) -> None:
    await _pause()
    result = await advanced_edit(
        image_path=str(generated_image),
        prompt="a plain light gray studio backdrop",
        edit_mode="background_swap",
        model=LITE_MODEL,
    )
    assert result["success"] is True, result.get("message")
    assert result["edit_mode"] == "background_swap"
    assert result["method"] == "generative"
    assert result["model"] == LITE_MODEL
    assert result["image_size"] == "1K"
    assert result["mask_used"] is False
    path = Path(result["output_path"])
    assert path.is_file()
    assert path.stat().st_size > 0
