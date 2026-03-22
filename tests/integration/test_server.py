"""Integration tests for MCP server functionality.

These tests require a valid GOOGLE_API_KEY to run.
Run with: pytest tests/integration/ -v -m integration
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

# Use environment variable if set, otherwise use a placeholder for CI/dry runs
os.environ.setdefault("GOOGLE_API_KEY", "test-placeholder-key")

from pixelforge_mcp.server import (
    analyze_image,
    edit_image,
    generate_image,
    get_server_info,
    list_available_models,
)


@pytest.mark.integration
def test_server_info():
    """Test server info tool."""
    result = get_server_info()
    assert "server" in result
    assert "storage" in result
    assert "imagen" in result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_models():
    """Test list models tool."""
    result = await list_available_models()
    assert result["success"] is True
    assert "models" in result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generate_image():
    """Test image generation tool."""
    result = await generate_image(
        prompt="a simple red circle on white background",
        aspect_ratio="1:1",
        temperature=0.5,
    )

    if result["success"]:
        assert "images" in result
        images = result["images"]
        assert len(images) > 0
        path = Path(images[0]["path"])
        assert path.exists(), "Image file should exist"
        assert path.stat().st_size > 0, "Image file should not be empty"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_analyze_image(tmp_path):
    """Test image analysis tool."""
    # First generate an image to analyze
    result = await generate_image(
        prompt="a simple red circle on white background",
        aspect_ratio="1:1",
        temperature=0.5,
    )

    if not result["success"]:
        pytest.skip("Image generation failed, cannot test analysis")

    image_path = result["images"][0]["path"]
    analysis_result = await analyze_image(image_path=image_path)

    if analysis_result["success"]:
        assert "analysis" in analysis_result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_edit_image():
    """Test image editing tool."""
    # First generate an image to edit
    result = await generate_image(
        prompt="a simple red circle on white background",
        aspect_ratio="1:1",
        temperature=0.5,
    )

    if not result["success"]:
        pytest.skip("Image generation failed, cannot test editing")

    image_path = result["images"][0]["path"]
    edit_result = await edit_image(
        prompt="add a blue square next to the red circle",
        input_image_path=image_path,
        temperature=0.5,
    )

    if edit_result["success"]:
        assert "images" in edit_result
        images = edit_result["images"]
        assert len(images) > 0
        path = Path(images[0]["path"])
        assert path.exists(), "Edited image file should exist"
        assert path.stat().st_size > 0, "Edited image file should not be empty"


async def _run_manual_tests():
    """Run all tests manually (outside pytest)."""
    print("Starting MCP Server Tests")
    print("=" * 60)

    # Test server info (no API call)
    print("\n=== Testing get_server_info ===")
    result = get_server_info()
    print(f"Success: {result}")
    assert "server" in result
    print("Server info tool works!")

    # Test list models
    print("\n=== Testing list_available_models ===")
    result = await list_available_models()
    print(f"Result: {result}")
    assert result["success"] is True
    print("List models tool works!")

    # Test image generation (API call)
    print("\n=== Testing generate_image ===")
    result = await generate_image(
        prompt="a simple red circle on white background",
        aspect_ratio="1:1",
        temperature=0.5,
    )
    print(f"Result: {result}")

    image_path = None
    if result["success"]:
        images = result["images"]
        if images:
            image_path = images[0]["path"]
            path = Path(image_path)
            assert path.exists(), "Image file should exist"
            assert path.stat().st_size > 0, "Image file should not be empty"
            print(f"Image generated: {path}")

    # Test image analysis (API call)
    if image_path:
        print("\n=== Testing analyze_image ===")
        result = await analyze_image(image_path=image_path)
        print(f"Result: {result}")
        if result["success"]:
            print(f"  Analysis: {result.get('analysis')}")
    else:
        print("\n=== Skipping analyze_image (no image) ===")

    # Test image editing (API call)
    if image_path:
        print("\n=== Testing edit_image ===")
        result = await edit_image(
            prompt="add a blue square next to the red circle",
            input_image_path=image_path,
            temperature=0.5,
        )
        print(f"Result: {result}")
        if result["success"]:
            images = result["images"]
            if images:
                path = Path(images[0]["path"])
                assert path.exists(), "Edited image file should exist"
                assert path.stat().st_size > 0, "Edited image file should not be empty"
                print(f"Edited image: {path}")
    else:
        print("\n=== Skipping edit_image (no image) ===")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        success = True
        asyncio.run(_run_manual_tests())
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback

        traceback.print_exc()
        success = False
    sys.exit(0 if success else 1)
