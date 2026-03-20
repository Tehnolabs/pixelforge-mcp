"""Unit tests for input validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from pixelforge_mcp.utils.validation import (
    ASPECT_RATIOS,
    FORMAT_EXTENSIONS,
    IMAGE_SIZES,
    OUTPUT_FORMATS,
    PERSON_GENERATION_OPTIONS,
    PERSON_GENERATION_SDK_MAP,
    AnalyzeImageInput,
    EditImageInput,
    GenerateImageInput,
)


class TestGenerateImageInput:
    """Tests for GenerateImageInput validation."""

    def test_valid_input_with_defaults(self):
        """Test valid input with default values."""
        input_data = GenerateImageInput(prompt="a beautiful sunset")

        assert input_data.prompt == "a beautiful sunset"
        assert input_data.output_filename is None
        assert input_data.aspect_ratio == "1:1"
        assert input_data.temperature == 0.7
        assert input_data.model is None
        assert input_data.safety_setting == "preset:strict"
        assert input_data.image_size is None
        assert input_data.number_of_images == 1
        assert input_data.output_format == "png"
        assert input_data.person_generation is None

    def test_valid_input_with_custom_values(self):
        """Test valid input with custom values."""
        input_data = GenerateImageInput(
            prompt="test prompt",
            output_filename="test.png",
            aspect_ratio="16:9",
            temperature=0.9,
            model="custom-model",
            safety_setting="preset:relaxed",
        )

        assert input_data.prompt == "test prompt"
        assert input_data.output_filename == "test.png"
        assert input_data.aspect_ratio == "16:9"
        assert input_data.temperature == 0.9
        assert input_data.model == "custom-model"
        assert input_data.safety_setting == "preset:relaxed"

    def test_empty_prompt_rejected(self):
        """Test empty prompt is rejected."""
        with pytest.raises(ValidationError, match="Prompt cannot be empty"):
            GenerateImageInput(prompt="")

    def test_whitespace_prompt_rejected(self):
        """Test whitespace-only prompt is rejected."""
        with pytest.raises(ValidationError, match="Prompt cannot be empty"):
            GenerateImageInput(prompt="   ")

    def test_long_prompt_rejected(self):
        """Test overly long prompt is rejected."""
        long_prompt = "a" * 2001
        with pytest.raises(ValidationError, match="Prompt too long"):
            GenerateImageInput(prompt=long_prompt)

    def test_max_length_prompt_accepted(self):
        """Test prompt at max length is accepted."""
        max_prompt = "a" * 2000
        input_data = GenerateImageInput(prompt=max_prompt)
        assert len(input_data.prompt) == 2000

    def test_panoramic_aspect_ratios_accepted(self):
        """Test panoramic aspect ratios are accepted by GenerateImageInput."""
        for ratio in ["1:4", "4:1", "1:8", "8:1"]:
            input_data = GenerateImageInput(prompt="test", aspect_ratio=ratio)
            assert input_data.aspect_ratio == ratio

    def test_invalid_aspect_ratio(self):
        """Test invalid aspect ratio is rejected."""
        with pytest.raises(ValidationError, match="Invalid aspect ratio"):
            GenerateImageInput(prompt="test", aspect_ratio="5:5")

    def test_all_aspect_ratios_valid(self):
        """Test all defined aspect ratios are accepted."""
        for ratio in ASPECT_RATIOS:
            input_data = GenerateImageInput(prompt="test", aspect_ratio=ratio)
            assert input_data.aspect_ratio == ratio

    def test_temperature_too_low(self):
        """Test temperature below 0 is rejected."""
        with pytest.raises(ValidationError):
            GenerateImageInput(prompt="test", temperature=-0.1)

    def test_temperature_too_high(self):
        """Test temperature above 2 is rejected."""
        with pytest.raises(ValidationError):
            GenerateImageInput(prompt="test", temperature=2.1)

    def test_temperature_boundaries(self):
        """Test temperature at boundaries is accepted."""
        # 0.0
        input_data = GenerateImageInput(prompt="test", temperature=0.0)
        assert input_data.temperature == 0.0

        # 2.0
        input_data = GenerateImageInput(prompt="test", temperature=2.0)
        assert input_data.temperature == 2.0

    def test_filename_with_path_separator_rejected(self):
        """Test filename with path separators is rejected."""
        with pytest.raises(ValidationError, match="cannot contain path separators"):
            GenerateImageInput(prompt="test", output_filename="../test.png")

        with pytest.raises(ValidationError, match="cannot contain path separators"):
            GenerateImageInput(prompt="test", output_filename="dir/test.png")

    def test_filename_with_invalid_characters_rejected(self):
        """Test filename with invalid characters is rejected."""
        with pytest.raises(ValidationError, match="can only contain"):
            GenerateImageInput(prompt="test", output_filename="test@file.png")

    def test_filename_auto_extension(self):
        """Test filename automatically gets .png extension if missing."""
        input_data = GenerateImageInput(prompt="test", output_filename="myimage")
        assert input_data.output_filename == "myimage.png"

    def test_filename_with_valid_extensions(self):
        """Test filenames with valid extensions are accepted."""
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            input_data = GenerateImageInput(prompt="test", output_filename=f"test{ext}")
            assert input_data.output_filename == f"test{ext}"

    def test_filename_with_mixed_case_extension(self):
        """Test filename with mixed case extension."""
        input_data = GenerateImageInput(prompt="test", output_filename="test.PNG")
        assert input_data.output_filename == "test.PNG"


class TestGenerateImageSize:
    """Tests for image_size validation."""

    def test_valid_image_sizes(self):
        """Test all valid image sizes are accepted."""
        for size in IMAGE_SIZES:
            input_data = GenerateImageInput(prompt="test", image_size=size)
            assert input_data.image_size == size

    def test_image_size_case_insensitive(self):
        """Test image_size accepts lowercase and normalizes to uppercase."""
        input_data = GenerateImageInput(prompt="test", image_size="4k")
        assert input_data.image_size == "4K"

        input_data = GenerateImageInput(prompt="test", image_size="2k")
        assert input_data.image_size == "2K"

    def test_invalid_image_size_rejected(self):
        """Test invalid image sizes are rejected."""
        with pytest.raises(ValidationError, match="Invalid image size"):
            GenerateImageInput(prompt="test", image_size="8K")

    def test_none_image_size_accepted(self):
        """Test None image_size is accepted (uses API default)."""
        input_data = GenerateImageInput(prompt="test", image_size=None)
        assert input_data.image_size is None

    def test_default_image_size_is_none(self):
        """Test default image_size is None (API uses 1K)."""
        input_data = GenerateImageInput(prompt="test")
        assert input_data.image_size is None


class TestGenerateNumberOfImages:
    """Tests for number_of_images validation."""

    def test_default_is_one(self):
        """Test default number_of_images is 1."""
        input_data = GenerateImageInput(prompt="test")
        assert input_data.number_of_images == 1

    def test_valid_range(self):
        """Test valid range 1-4 is accepted."""
        for n in [1, 2, 3, 4]:
            input_data = GenerateImageInput(prompt="test", number_of_images=n)
            assert input_data.number_of_images == n

    def test_zero_rejected(self):
        """Test 0 images is rejected."""
        with pytest.raises(ValidationError):
            GenerateImageInput(prompt="test", number_of_images=0)

    def test_five_rejected(self):
        """Test 5 images is rejected (max is 4)."""
        with pytest.raises(ValidationError):
            GenerateImageInput(prompt="test", number_of_images=5)

    def test_negative_rejected(self):
        """Test negative number is rejected."""
        with pytest.raises(ValidationError):
            GenerateImageInput(prompt="test", number_of_images=-1)


class TestGenerateOutputFormat:
    """Tests for output_format validation."""

    def test_default_is_png(self):
        """Test default output format is png."""
        input_data = GenerateImageInput(prompt="test")
        assert input_data.output_format == "png"

    def test_all_valid_formats(self):
        """Test all valid formats are accepted."""
        for fmt in OUTPUT_FORMATS:
            input_data = GenerateImageInput(prompt="test", output_format=fmt)
            assert input_data.output_format == fmt

    def test_format_case_insensitive(self):
        """Test format is case insensitive."""
        input_data = GenerateImageInput(prompt="test", output_format="JPEG")
        assert input_data.output_format == "jpeg"

        input_data = GenerateImageInput(prompt="test", output_format="WebP")
        assert input_data.output_format == "webp"

    def test_invalid_format_rejected(self):
        """Test invalid formats are rejected."""
        with pytest.raises(ValidationError, match="Invalid output format"):
            GenerateImageInput(prompt="test", output_format="bmp")

        with pytest.raises(ValidationError, match="Invalid output format"):
            GenerateImageInput(prompt="test", output_format="gif")


class TestGeneratePersonGeneration:
    """Tests for person_generation validation."""

    def test_default_is_none(self):
        """Test default person_generation is None."""
        input_data = GenerateImageInput(prompt="test")
        assert input_data.person_generation is None

    def test_all_valid_options(self):
        """Test all valid person_generation options."""
        for option in PERSON_GENERATION_OPTIONS:
            input_data = GenerateImageInput(prompt="test", person_generation=option)
            assert input_data.person_generation == option

    def test_case_insensitive(self):
        """Test person_generation is case insensitive."""
        input_data = GenerateImageInput(prompt="test", person_generation="ALLOW_ALL")
        assert input_data.person_generation == "allow_all"

        input_data = GenerateImageInput(prompt="test", person_generation="Allow_Adult")
        assert input_data.person_generation == "allow_adult"

    def test_invalid_option_rejected(self):
        """Test invalid person_generation is rejected."""
        with pytest.raises(ValidationError, match="Invalid person generation"):
            GenerateImageInput(prompt="test", person_generation="block_all")

    def test_sdk_map_covers_all_options(self):
        """Test SDK map has entries for all user-facing options."""
        for option in PERSON_GENERATION_OPTIONS:
            assert option in PERSON_GENERATION_SDK_MAP


class TestGenerateAllNewParams:
    """Tests for using all new params together."""

    def test_all_new_params_combined(self):
        """Test all new Phase 1 params work together."""
        input_data = GenerateImageInput(
            prompt="test prompt",
            image_size="4K",
            number_of_images=3,
            output_format="webp",
            person_generation="allow_all",
        )
        assert input_data.image_size == "4K"
        assert input_data.number_of_images == 3
        assert input_data.output_format == "webp"
        assert input_data.person_generation == "allow_all"


class TestEditImageInput:
    """Tests for EditImageInput validation."""

    def test_valid_input(self, tmp_path):
        """Test valid edit input."""
        image_path = tmp_path / "test.png"
        image_path.touch()

        input_data = EditImageInput(
            prompt="add clouds",
            input_image_path=str(image_path),
            temperature=0.8,
        )

        assert input_data.prompt == "add clouds"
        assert Path(input_data.input_image_path) == image_path.absolute()
        assert input_data.temperature == 0.8
        assert input_data.output_filename is None

    def test_empty_prompt_rejected(self):
        """Test empty prompt is rejected."""
        with pytest.raises(ValidationError, match="Prompt cannot be empty"):
            EditImageInput(prompt="", input_image_path="/test.png")

    def test_long_prompt_rejected(self):
        """Test overly long prompt is rejected."""
        long_prompt = "a" * 2001
        with pytest.raises(ValidationError, match="Prompt too long"):
            EditImageInput(prompt=long_prompt, input_image_path="/test.png")

    def test_nonexistent_image_rejected(self):
        """Test non-existent input image is rejected."""
        with pytest.raises(ValidationError, match="not found"):
            EditImageInput(prompt="test", input_image_path="/nonexistent/image.png")

    def test_directory_path_rejected(self, tmp_path):
        """Test directory path is rejected as input image."""
        with pytest.raises(ValidationError, match="not a file"):
            EditImageInput(prompt="test", input_image_path=str(tmp_path))

    def test_invalid_image_format_rejected(self, tmp_path):
        """Test invalid image format is rejected."""
        text_file = tmp_path / "test.txt"
        text_file.touch()

        with pytest.raises(ValidationError, match="Invalid image format"):
            EditImageInput(prompt="test", input_image_path=str(text_file))

    def test_temperature_validation(self, tmp_path):
        """Test temperature validation in edit input."""
        image_path = tmp_path / "test.png"
        image_path.touch()

        with pytest.raises(ValidationError):
            EditImageInput(
                prompt="test",
                input_image_path=str(image_path),
                temperature=-0.1,
            )

        with pytest.raises(ValidationError):
            EditImageInput(
                prompt="test",
                input_image_path=str(image_path),
                temperature=2.1,
            )

    def test_model_param_accepted(self, tmp_path):
        """Test model parameter is accepted for edit."""
        image_path = tmp_path / "test.png"
        image_path.touch()

        input_data = EditImageInput(
            prompt="add clouds",
            input_image_path=str(image_path),
            model="gemini-3-pro-image-preview",
        )
        assert input_data.model == "gemini-3-pro-image-preview"

    def test_model_default_is_none(self, tmp_path):
        """Test model default is None."""
        image_path = tmp_path / "test.png"
        image_path.touch()

        input_data = EditImageInput(
            prompt="add clouds",
            input_image_path=str(image_path),
        )
        assert input_data.model is None

    def test_output_format_accepted(self, tmp_path):
        """Test output_format parameter for edit."""
        image_path = tmp_path / "test.png"
        image_path.touch()

        for fmt in OUTPUT_FORMATS:
            input_data = EditImageInput(
                prompt="add clouds",
                input_image_path=str(image_path),
                output_format=fmt,
            )
            assert input_data.output_format == fmt

    def test_output_format_default_is_png(self, tmp_path):
        """Test output_format default is png."""
        image_path = tmp_path / "test.png"
        image_path.touch()

        input_data = EditImageInput(
            prompt="add clouds",
            input_image_path=str(image_path),
        )
        assert input_data.output_format == "png"

    def test_invalid_output_format_rejected(self, tmp_path):
        """Test invalid output format is rejected for edit."""
        image_path = tmp_path / "test.png"
        image_path.touch()

        with pytest.raises(ValidationError, match="Invalid output format"):
            EditImageInput(
                prompt="add clouds",
                input_image_path=str(image_path),
                output_format="bmp",
            )


class TestAnalyzeImageInput:
    """Tests for AnalyzeImageInput validation."""

    def test_valid_input(self, tmp_path):
        """Test valid analyze input."""
        image_path = tmp_path / "test.jpg"
        image_path.touch()

        input_data = AnalyzeImageInput(image_path=str(image_path))

        assert Path(input_data.image_path) == image_path.absolute()

    def test_nonexistent_image_rejected(self):
        """Test non-existent image is rejected."""
        with pytest.raises(ValidationError, match="not found"):
            AnalyzeImageInput(image_path="/nonexistent/image.png")

    def test_directory_path_rejected(self, tmp_path):
        """Test directory path is rejected."""
        with pytest.raises(ValidationError, match="not a file"):
            AnalyzeImageInput(image_path=str(tmp_path))

    def test_invalid_format_rejected(self, tmp_path):
        """Test invalid image format is rejected."""
        text_file = tmp_path / "test.txt"
        text_file.touch()

        with pytest.raises(ValidationError, match="Invalid image format"):
            AnalyzeImageInput(image_path=str(text_file))

    def test_all_valid_formats_accepted(self, tmp_path):
        """Test all valid image formats are accepted."""
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            image_path = tmp_path / f"test{ext}"
            image_path.touch()

            input_data = AnalyzeImageInput(image_path=str(image_path))
            assert Path(input_data.image_path).suffix.lower() == ext

    def test_none_prompt_accepted(self, tmp_path):
        """Test None prompt is accepted (uses default)."""
        image_path = tmp_path / "test.png"
        image_path.touch()

        input_data = AnalyzeImageInput(image_path=str(image_path), prompt=None)
        assert input_data.prompt is None

    def test_valid_custom_prompt(self, tmp_path):
        """Test valid custom prompt is accepted."""
        image_path = tmp_path / "test.png"
        image_path.touch()

        input_data = AnalyzeImageInput(
            image_path=str(image_path),
            prompt="Extract all text from this image",
        )
        assert input_data.prompt == "Extract all text from this image"

    def test_prompt_whitespace_stripped(self, tmp_path):
        """Test prompt whitespace is stripped."""
        image_path = tmp_path / "test.png"
        image_path.touch()

        input_data = AnalyzeImageInput(
            image_path=str(image_path), prompt="  some prompt  "
        )
        assert input_data.prompt == "some prompt"

    def test_empty_prompt_rejected(self, tmp_path):
        """Test empty prompt string is rejected."""
        image_path = tmp_path / "test.png"
        image_path.touch()

        with pytest.raises(ValidationError, match="Prompt cannot be empty"):
            AnalyzeImageInput(image_path=str(image_path), prompt="")

    def test_whitespace_only_prompt_rejected(self, tmp_path):
        """Test whitespace-only prompt is rejected."""
        image_path = tmp_path / "test.png"
        image_path.touch()

        with pytest.raises(ValidationError, match="Prompt cannot be empty"):
            AnalyzeImageInput(image_path=str(image_path), prompt="   ")

    def test_long_prompt_rejected(self, tmp_path):
        """Test prompt over 2000 characters is rejected."""
        image_path = tmp_path / "test.png"
        image_path.touch()

        with pytest.raises(ValidationError, match="Prompt too long"):
            AnalyzeImageInput(image_path=str(image_path), prompt="a" * 2001)

    def test_max_length_prompt_accepted(self, tmp_path):
        """Test prompt at exactly 2000 characters is accepted."""
        image_path = tmp_path / "test.png"
        image_path.touch()

        input_data = AnalyzeImageInput(image_path=str(image_path), prompt="a" * 2000)
        assert len(input_data.prompt) == 2000


class TestAspectRatios:
    """Tests for aspect ratio constants."""

    def test_aspect_ratios_defined(self):
        """Test aspect ratios constant is defined."""
        assert isinstance(ASPECT_RATIOS, list)
        assert len(ASPECT_RATIOS) > 0

    def test_common_ratios_included(self):
        """Test common aspect ratios are included."""
        common_ratios = ["1:1", "16:9", "9:16", "4:3", "3:4"]
        for ratio in common_ratios:
            assert ratio in ASPECT_RATIOS, f"{ratio} should be in ASPECT_RATIOS"

    def test_panoramic_ratios_included(self):
        """Test panoramic aspect ratios are included."""
        panoramic_ratios = ["1:4", "4:1", "1:8", "8:1"]
        for ratio in panoramic_ratios:
            assert ratio in ASPECT_RATIOS, f"{ratio} should be in ASPECT_RATIOS"


class TestConstants:
    """Tests for validation constants."""

    def test_format_extensions_complete(self):
        """Test all output formats have an extension mapping."""
        for fmt in OUTPUT_FORMATS:
            assert fmt in FORMAT_EXTENSIONS

    def test_format_extensions_have_dot(self):
        """Test all extensions start with a dot."""
        for ext in FORMAT_EXTENSIONS.values():
            assert ext.startswith(".")

    def test_image_sizes_are_uppercase(self):
        """Test all image sizes are uppercase."""
        for size in IMAGE_SIZES:
            assert size == size.upper()

    def test_person_generation_sdk_map_values(self):
        """Test SDK map values are uppercase SDK format."""
        for sdk_val in PERSON_GENERATION_SDK_MAP.values():
            assert sdk_val == sdk_val.upper()
            assert sdk_val.startswith("ALLOW_") or sdk_val == "ALLOW_NONE"
