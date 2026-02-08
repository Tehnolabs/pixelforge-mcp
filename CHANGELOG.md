# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-08

### Added

- Image generation from text prompts via `generate_image` tool
- Image editing with text prompts via `edit_image` tool
- AI-powered image analysis via `analyze_image` tool
- Model listing with capabilities and selection guidance via `list_available_models` tool
- Server configuration and status via `get_server_info` tool
- Per-request model switching between gemini-2.5-flash-image and gemini-3-pro-image-preview
- 10 aspect ratio options (1:1, 16:9, 9:16, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 21:9)
- Temperature control for creativity adjustment (0.0-1.0)
- Pydantic-based configuration with YAML file support
- Environment variable support (GOOGLE_API_KEY, GOOGLE_GENERATIVE_AI_API_KEY, GEMINI_API_KEY)
- 72 unit tests with full coverage
- PyPI package distribution via pipx

[Unreleased]: https://github.com/tehnolabs/pixelforge-mcp/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tehnolabs/pixelforge-mcp/releases/tag/v0.1.0
