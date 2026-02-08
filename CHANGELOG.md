# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.5] - 2026-02-09

### Added

- Custom `prompt` parameter for `analyze_image` tool — directs analysis to focus on specific aspects (OCR, accessibility, colors, etc.)
- 8 new unit tests for prompt validation and custom prompt handling

## [0.1.4] - 2026-02-08

### Fixed

- Hero image broken on PyPI — use absolute GitHub URL instead of relative path

## [0.1.3] - 2026-02-08

### Added

- Hero image for README (AI-generated pixel forge anvil)
- Whitelisted docs/assets/ in .gitignore for repo images

## [0.1.2] - 2026-02-08

### Added

- MCP install instructions for Cursor, VS Code, Windsurf, and Kiro
- Updated troubleshooting sections with config paths for all 6 supported clients

## [0.1.1] - 2026-02-08

### Changed

- Rewrote README.md with PyPI installation, tool documentation, and badges
- Rewrote CLAUDE.md as a concise AI agent integration guide
- Rewrote docs/configuration.md with PyPI-first install and 3 configuration methods
- Added CONTRIBUTING.md with development setup, coding standards, and PR process
- Added CHANGELOG.md following Keep a Changelog format
- Fixed version references from 1.0.0 to 0.1.0

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

[Unreleased]: https://github.com/tehnolabs/pixelforge-mcp/compare/v0.1.5...HEAD
[0.1.5]: https://github.com/tehnolabs/pixelforge-mcp/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/tehnolabs/pixelforge-mcp/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/tehnolabs/pixelforge-mcp/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/tehnolabs/pixelforge-mcp/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/tehnolabs/pixelforge-mcp/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/tehnolabs/pixelforge-mcp/releases/tag/v0.1.0
