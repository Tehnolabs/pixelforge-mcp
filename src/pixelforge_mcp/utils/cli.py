"""CLI execution utilities for gemini-imagen."""

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CLIResult:
    """Result from CLI command execution."""

    success: bool
    output: str
    error: Optional[str]
    return_code: int
    data: Optional[Dict[str, Any]] = None  # Parsed JSON data if available


class ImagenCLI:
    """Wrapper for gemini-imagen CLI commands."""

    def __init__(self, timeout: int = 120):
        """Initialize CLI wrapper.

        Args:
            timeout: Command timeout in seconds
        """
        self.timeout = timeout
        self._verify_cli_available()

    def _verify_cli_available(self) -> None:
        """Verify gemini-imagen CLI is installed."""
        try:
            result = subprocess.run(
                ["imagen", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise RuntimeError("gemini-imagen CLI not working properly")
            logger.info(f"gemini-imagen CLI verified: {result.stdout.strip()}")
        except FileNotFoundError:
            raise RuntimeError(
                "gemini-imagen CLI not found. Install with: pip install gemini-imagen"
            )

    def _run_command(
        self,
        args: List[str],
        parse_json: bool = False,
    ) -> CLIResult:
        """Execute a CLI command.

        Args:
            args: Command arguments (without 'imagen' prefix)
            parse_json: Whether to parse output as JSON

        Returns:
            CLIResult with command output
        """
        command = ["imagen"] + args
        logger.debug(f"Executing: {' '.join(command)}")

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            output = result.stdout.strip()
            error = result.stderr.strip() if result.returncode != 0 else None

            # Parse JSON if requested
            data = None
            if parse_json and output:
                try:
                    data = json.loads(output)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON output: {e}")

            return CLIResult(
                success=result.returncode == 0,
                output=output,
                error=error,
                return_code=result.returncode,
                data=data,
            )

        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {self.timeout}s")
            return CLIResult(
                success=False,
                output="",
                error=f"Command timed out after {self.timeout}s",
                return_code=-1,
            )
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return CLIResult(
                success=False,
                output="",
                error=str(e),
                return_code=-1,
            )

    def generate(
        self,
        prompt: str,
        output_path: Path,
        aspect_ratio: Optional[str] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        safety_setting: Optional[str] = None,
    ) -> CLIResult:
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the image
            output_path: Where to save the generated image
            aspect_ratio: Image aspect ratio (e.g., "16:9")
            temperature: Generation temperature (0-1)
            model: Model to use for generation
            safety_setting: Safety filter setting

        Returns:
            CLIResult with generation details
        """
        args = ["generate", prompt, "-o", str(output_path), "--json"]

        if aspect_ratio:
            args.extend(["--aspect-ratio", aspect_ratio])
        if temperature is not None:
            args.extend(["--temperature", str(temperature)])
        if model:
            args.extend(["--model", model])
        if safety_setting:
            args.extend(["--safety-setting", safety_setting])

        return self._run_command(args, parse_json=True)

    def edit(
        self,
        prompt: str,
        input_path: Path,
        output_path: Path,
        temperature: Optional[float] = None,
    ) -> CLIResult:
        """Edit an existing image with a text prompt.

        Args:
            prompt: Text description of desired changes
            input_path: Path to the input image
            output_path: Where to save the edited image
            temperature: Generation temperature (0-1)

        Returns:
            CLIResult with editing details
        """
        args = [
            "generate",
            prompt,
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "--json",
        ]

        if temperature is not None:
            args.extend(["--temperature", str(temperature)])

        return self._run_command(args, parse_json=True)

    def analyze(self, image_path: Path) -> CLIResult:
        """Analyze an image and get a description.

        Args:
            image_path: Path to the image to analyze

        Returns:
            CLIResult with analysis data
        """
        args = ["analyze", str(image_path), "--json"]
        return self._run_command(args, parse_json=True)

    def list_models(self) -> CLIResult:
        """List available generation models.

        Returns:
            CLIResult with model list
        """
        args = ["models", "list", "--json"]
        return self._run_command(args, parse_json=True)
