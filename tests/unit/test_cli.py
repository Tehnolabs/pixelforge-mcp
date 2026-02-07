"""Unit tests for CLI execution utilities."""

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pixelforge_mcp.utils.cli import CLIResult, ImagenCLI


class TestCLIResult:
    """Tests for CLIResult dataclass."""

    def test_create_successful_result(self):
        """Test creating a successful CLIResult."""
        result = CLIResult(
            success=True,
            output="Image generated",
            error=None,
            return_code=0,
            data={"path": "/test.png"}
        )
        assert result.success is True
        assert result.output == "Image generated"
        assert result.error is None
        assert result.return_code == 0
        assert result.data == {"path": "/test.png"}

    def test_create_failed_result(self):
        """Test creating a failed CLIResult."""
        result = CLIResult(
            success=False,
            output="",
            error="Command failed",
            return_code=1,
        )
        assert result.success is False
        assert result.error == "Command failed"
        assert result.return_code == 1
        assert result.data is None


class TestImagenCLI:
    """Tests for ImagenCLI wrapper."""

    @patch("subprocess.run")
    def test_init_verifies_cli(self, mock_run):
        """Test initialization verifies CLI is available."""
        # Mock successful version check
        mock_run.return_value = Mock(
            returncode=0,
            stdout="imagen 0.6.6",
            stderr=""
        )

        cli = ImagenCLI()
        assert cli.timeout == 120

        # Verify version check was called
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "imagen"
        assert "--version" in args

    @patch("subprocess.run")
    def test_init_raises_if_cli_not_found(self, mock_run):
        """Test initialization raises if CLI not found."""
        mock_run.side_effect = FileNotFoundError()

        with pytest.raises(RuntimeError, match="gemini-imagen CLI not found"):
            ImagenCLI()

    @patch("subprocess.run")
    def test_init_raises_if_cli_not_working(self, mock_run):
        """Test initialization raises if CLI returns error."""
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="error")

        with pytest.raises(RuntimeError, match="not working properly"):
            ImagenCLI()

    @patch("subprocess.run")
    def test_generate_success(self, mock_run, tmp_path):
        """Test successful image generation."""
        # Mock CLI verification
        mock_run.return_value = Mock(
            returncode=0,
            stdout="imagen 0.6.6",
            stderr=""
        )
        cli = ImagenCLI()

        # Mock generate command
        output_data = {"status": "success", "path": "/test.png"}
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps(output_data),
            stderr=""
        )

        output_path = tmp_path / "test.png"
        result = cli.generate(
            prompt="test prompt",
            output_path=output_path,
            aspect_ratio="16:9",
            temperature=0.8
        )

        assert result.success is True
        assert result.return_code == 0
        assert result.data == output_data

        # Verify command was called with correct args
        call_args = mock_run.call_args[0][0]
        assert "imagen" in call_args
        assert "generate" in call_args
        assert "test prompt" in call_args
        assert "--aspect-ratio" in call_args
        assert "16:9" in call_args
        assert "--temperature" in call_args
        assert "0.8" in call_args

    @patch("subprocess.run")
    def test_generate_failure(self, mock_run):
        """Test failed image generation."""
        # Mock CLI verification
        mock_run.return_value = Mock(
            returncode=0,
            stdout="imagen 0.6.6",
            stderr=""
        )
        cli = ImagenCLI()

        # Mock failed generate command
        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="API key invalid"
        )

        result = cli.generate(
            prompt="test",
            output_path=Path("/test.png")
        )

        assert result.success is False
        assert result.return_code == 1
        assert "API key invalid" in result.error

    @patch("subprocess.run")
    def test_generate_timeout(self, mock_run):
        """Test generation with timeout."""
        # Mock CLI verification
        mock_run.return_value = Mock(
            returncode=0,
            stdout="imagen 0.6.6",
            stderr=""
        )
        cli = ImagenCLI(timeout=5)

        # Mock timeout
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd="imagen",
            timeout=5
        )

        result = cli.generate(
            prompt="test",
            output_path=Path("/test.png")
        )

        assert result.success is False
        assert result.return_code == -1
        assert "timed out" in result.error.lower()

    @patch("subprocess.run")
    def test_edit_success(self, mock_run, tmp_path):
        """Test successful image editing."""
        # Mock CLI verification
        mock_run.return_value = Mock(
            returncode=0,
            stdout="imagen 0.6.6",
            stderr=""
        )
        cli = ImagenCLI()

        # Create test input file
        input_path = tmp_path / "input.png"
        input_path.touch()
        output_path = tmp_path / "output.png"

        # Mock edit command
        output_data = {"status": "success"}
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps(output_data),
            stderr=""
        )

        result = cli.edit(
            prompt="add clouds",
            input_path=input_path,
            output_path=output_path,
            temperature=0.7
        )

        assert result.success is True
        assert result.data == output_data

        # Verify command includes input flag
        call_args = mock_run.call_args[0][0]
        assert "-i" in call_args
        assert str(input_path) in call_args

    @patch("subprocess.run")
    def test_analyze_success(self, mock_run, tmp_path):
        """Test successful image analysis."""
        # Mock CLI verification
        mock_run.return_value = Mock(
            returncode=0,
            stdout="imagen 0.6.6",
            stderr=""
        )
        cli = ImagenCLI()

        # Create test image
        image_path = tmp_path / "test.png"
        image_path.touch()

        # Mock analyze command
        analysis_data = {"description": "A beautiful sunset"}
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps(analysis_data),
            stderr=""
        )

        result = cli.analyze(image_path)

        assert result.success is True
        assert result.data == analysis_data

        # Verify command
        call_args = mock_run.call_args[0][0]
        assert "analyze" in call_args
        assert str(image_path) in call_args

    @patch("subprocess.run")
    def test_list_models_success(self, mock_run):
        """Test successful model listing."""
        # Mock CLI verification
        mock_run.return_value = Mock(
            returncode=0,
            stdout="imagen 0.6.6",
            stderr=""
        )
        cli = ImagenCLI()

        # Mock list models command
        models_data = {"models": ["gemini-2.5-flash-image"]}
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps(models_data),
            stderr=""
        )

        result = cli.list_models()

        assert result.success is True
        assert result.data == models_data

        # Verify command
        call_args = mock_run.call_args[0][0]
        assert "models" in call_args
        assert "list" in call_args

    @patch("subprocess.run")
    def test_json_parsing_failure(self, mock_run):
        """Test handling of invalid JSON output."""
        # Mock CLI verification
        mock_run.return_value = Mock(
            returncode=0,
            stdout="imagen 0.6.6",
            stderr=""
        )
        cli = ImagenCLI()

        # Mock command with invalid JSON
        mock_run.return_value = Mock(
            returncode=0,
            stdout="not valid json",
            stderr=""
        )

        result = cli.generate(
            prompt="test",
            output_path=Path("/test.png")
        )

        assert result.success is True
        assert result.data is None  # JSON parsing failed
        assert result.output == "not valid json"

    @patch("subprocess.run")
    def test_custom_timeout(self, mock_run):
        """Test CLI with custom timeout."""
        # Mock CLI verification
        mock_run.return_value = Mock(
            returncode=0,
            stdout="imagen 0.6.6",
            stderr=""
        )

        cli = ImagenCLI(timeout=300)
        assert cli.timeout == 300

        # Mock generate command
        mock_run.return_value = Mock(
            returncode=0,
            stdout="{}",
            stderr=""
        )

        cli.generate(prompt="test", output_path=Path("/test.png"))

        # Verify timeout was used
        assert mock_run.call_args[1]["timeout"] == 300
