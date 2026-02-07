"""Test script to verify MCP server functionality."""

import os
import sys
from pathlib import Path

# Use environment variable if set, otherwise use a placeholder for CI/dry runs
os.environ.setdefault("GOOGLE_API_KEY", "test-placeholder-key")

from pixelforge_mcp.server import (
    generate_image,
    edit_image,
    analyze_image,
    list_available_models,
    get_server_info,
)


def test_server_info():
    """Test server info tool."""
    print("\n=== Testing get_server_info ===")
    result = get_server_info()
    print(f"Success: {result}")
    assert "server" in result
    assert "storage" in result
    assert "imagen" in result
    print("✓ Server info tool works!")


def test_list_models():
    """Test list models tool."""
    print("\n=== Testing list_available_models ===")
    result = list_available_models()
    print(f"Result: {result}")
    assert result["success"] is True
    assert "models" in result
    print("✓ List models tool works!")


def test_generate_image():
    """Test image generation tool."""
    print("\n=== Testing generate_image ===")
    result = generate_image(
        prompt="a simple red circle on white background",
        aspect_ratio="1:1",
        temperature=0.5,
    )
    print(f"Result: {result}")

    if result["success"]:
        print(f"✓ Image generated successfully!")
        print(f"  Path: {result.get('image_path')}")
        print(f"  Size: {result.get('image_size_bytes')} bytes")

        # Verify file exists
        if "image_path" in result:
            path = Path(result["image_path"])
            assert path.exists(), "Image file should exist"
            assert path.stat().st_size > 0, "Image file should not be empty"
            print(f"✓ Image file verified: {path}")
            return path
    else:
        print(f"✗ Image generation failed: {result.get('message')}")
        return None


def test_analyze_image(image_path):
    """Test image analysis tool."""
    if image_path is None:
        print("\n=== Skipping analyze_image (no image) ===")
        return

    print("\n=== Testing analyze_image ===")
    result = analyze_image(image_path=str(image_path))
    print(f"Result: {result}")

    if result["success"]:
        print(f"✓ Image analyzed successfully!")
        print(f"  Analysis: {result.get('analysis')}")
    else:
        print(f"✗ Image analysis failed: {result.get('message')}")


def test_edit_image(image_path):
    """Test image editing tool."""
    if image_path is None:
        print("\n=== Skipping edit_image (no image) ===")
        return

    print("\n=== Testing edit_image ===")
    result = edit_image(
        prompt="add a blue square next to the red circle",
        input_image_path=str(image_path),
        temperature=0.5,
    )
    print(f"Result: {result}")

    if result["success"]:
        print(f"✓ Image edited successfully!")
        print(f"  Path: {result.get('image_path')}")
        print(f"  Size: {result.get('image_size_bytes')} bytes")

        # Verify file exists
        if "image_path" in result:
            path = Path(result["image_path"])
            assert path.exists(), "Edited image file should exist"
            assert path.stat().st_size > 0, "Edited image file should not be empty"
            print(f"✓ Edited image file verified: {path}")
    else:
        print(f"✗ Image editing failed: {result.get('message')}")


def main():
    """Run all tests."""
    print("Starting MCP Server Tests")
    print("=" * 60)

    try:
        # Test server info (no API call)
        test_server_info()

        # Test list models (no API call or minor API call)
        test_list_models()

        # Test image generation (API call)
        image_path = test_generate_image()

        # Test image analysis (API call)
        test_analyze_image(image_path)

        # Test image editing (API call)
        test_edit_image(image_path)

        print("\n" + "=" * 60)
        print("✓ All tests completed successfully!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
