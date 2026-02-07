"""Simple integration test to verify image generation works."""

import os
import sys
from pathlib import Path

# Use environment variable if set, otherwise use a placeholder for CI/dry runs
os.environ.setdefault("GOOGLE_API_KEY", "test-placeholder-key")

from pixelforge_mcp.config import get_config
from pixelforge_mcp.utils.cli import ImagenCLI


def test_direct_image_generation():
    """Test image generation directly through CLI wrapper."""
    print("\n" + "=" * 60)
    print("INTEGRATION TEST: Direct Image Generation")
    print("=" * 60)

    try:
        # Initialize config and CLI
        config = get_config()
        print(f"\n✓ Config loaded")
        print(f"  Output dir: {config.storage.output_dir}")

        cli = ImagenCLI(timeout=60)
        print(f"✓ CLI wrapper initialized")

        # Create output path
        output_path = config.storage.output_dir / "test_integration.png"
        print(f"\n→ Generating image: {output_path}")

        # Generate image
        result = cli.generate(
            prompt="a simple red circle on white background",
            output_path=output_path,
            aspect_ratio="1:1",
            temperature=0.5,
        )

        if result.success:
            print(f"✓ Image generation SUCCESSFUL!")
            print(f"  Return code: {result.return_code}")
            print(f"  Output: {result.output[:100] if result.output else 'N/A'}")

            # Verify file exists
            if output_path.exists():
                size = output_path.stat().st_size
                print(f"✓ Image file verified:")
                print(f"  Path: {output_path.absolute()}")
                print(f"  Size: {size:,} bytes")

                if size > 0:
                    print("\n" + "=" * 60)
                    print("✓✓✓ IMAGE_GENERATED_SUCCESSFULLY ✓✓✓")
                    print("=" * 60)
                    return True
                else:
                    print(f"✗ Error: Image file is empty")
                    return False
            else:
                print(f"✗ Error: Image file not found at {output_path}")
                return False
        else:
            print(f"✗ Image generation FAILED")
            print(f"  Error: {result.error}")
            print(f"  Return code: {result.return_code}")
            return False

    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run the integration test."""
    success = test_direct_image_generation()

    if success:
        print("\n✅ Integration test PASSED")
        return 0
    else:
        print("\n❌ Integration test FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
