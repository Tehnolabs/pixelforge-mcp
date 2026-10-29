# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0] - 2026-10-29

### Added

- Vertex AI backend for every tool: set `GOOGLE_CLOUD_PROJECT` with Application Default Credentials (`GOOGLE_APPLICATION_CREDENTIALS` for a service account); `GOOGLE_GENAI_USE_VERTEXAI=true` (or `GOOGLE_GENAI_USE_ENTERPRISE=true`) prefers it over an API key
- `model` override on `upscale_image` and `advanced_edit`
- Retry with exponential backoff on 408/429/5xx responses
- `backend` block in `get_server_info`
- ADR-003 documenting the Vertex AI backend and generative upscale

### Changed

- `upscale_image` and `advanced_edit` re-render with a Gemini image model (`gemini-nano-banana-2.1` by default) instead of the retired Imagen models; responses carry `model`, `method`, `image_size`, and `warnings`
- `upscale_image` outputs the smallest of 1K/2K/4K that covers the source times the factor, capped at 4K; `gemini-3.1-flash-lite-image` (1K only) is rejected
- `advanced_edit` passes an optional mask to the model as guidance; the edited region is not pixel-exact
- Vertex AI default location is `global`; `GOOGLE_CLOUD_LOCATION` overrides it
- `0.5K` is sent to the API as `512px`
- `generate_image` rejects image sizes the chosen model cannot output, including through quality presets (`quality="fast"` with 2K or 4K)
- `upscale_image` and `advanced_edit` use the nearest aspect ratio the model supports; models without panoramic output get a standard ratio and a warning
- API errors keep Google's text: 429 and 403 responses name the quota or permission problem, and a Vertex AI 404 at `global` points at the model ID rather than the location
- The Gemini API client is created with `vertexai=False`, so the SDK's `GOOGLE_GENAI_USE_VERTEXAI` / `GOOGLE_GENAI_USE_ENTERPRISE` switches cannot reroute an API key
- CI actions updated to v7; the integration job masks the project number and runs pytest with `--show-capture=no`
- The `vertex` extra is empty; no extra packages are needed for Vertex AI

### Fixed

- `upscale_image` and `advanced_edit` returned 404 because the Imagen models they relied on were retired by Google
- `TypeError` when a response has no parts, for example after an image safety stop; the error now names the finish reason
- Undecodable image bytes in a response are reported as such instead of surfacing a PIL error
- Vertex AI "Publisher model ... was not found" errors (either casing) now suggest `GOOGLE_CLOUD_LOCATION=global` for regional locations

## [0.6.0] - 2026-10-28

### Added

- Stable Gemini image models: `gemini-nano-banana-2.1` (new default), `gemini-3.1-flash-lite-image`, `gemini-3.1-flash-image`, `gemini-3-pro-image`
- `0.5K` image size (`gemini-3.1-flash-image` only)
- `model` override on `analyze_image`, `extract_text`, `detect_objects`, `compare_images`, and `optimize_prompt`
- Per-size pricing and a separate `analysis_models` list in `list_available_models`
- CI workflow running ruff, black, mypy, and pytest on Python 3.10, 3.12, and 3.14
- ADR-002 documenting the model migration

### Changed

- Quality presets: `fast` uses `gemini-3.1-flash-lite-image`, `balanced` uses `gemini-nano-banana-2.1` at 1K, `quality` uses `gemini-3-pro-image` at 2K
- Analysis tools default to `gemini-3.5-flash-lite`; `gemini-2.5-flash` is no longer available to new Gemini API users
- Retired preview IDs `gemini-3-pro-image-preview` and `gemini-3.1-flash-image-preview` are aliased to their stable replacements with a response warning
- `estimate_cost` without a model lists only models that price the requested operation
- `person_generation` is accepted but documented as not enforced by the Gemini API
- Dependency ranges: `google-genai>=1.62.0,<3.0.0`, `fastmcp>=3.1.0,<5.0.0`, new `certifi>=2026.10.19` floor (eMudhra/emSign CA incident)
- Tool count corrected to 19 in the documentation

### Removed

- Imagen 4 models (`imagen-4.0-generate-001`, `imagen-4.0-fast-generate-001`, `imagen-4.0-ultra-generate-001`), shut down by Google on 2026-08-17. Requests now return an error naming the replacement

### Fixed

- `advanced_edit` failed with a `ValidationError` whenever a mask was supplied, and sent empty reference images
- `upscale_image` sent an empty image to Vertex AI
- All 53 `mypy --strict` errors

## [0.5.4] - 2026-03-30

### Fixed

- Pyright type errors in Pydantic config models — `Field(default=...)` keyword form and lambda-wrapped `default_factory`

## [0.5.3] - 2026-03-30

### Changed

- `person_generation` on Gemini models is now warn-only: the chosen model is always preserved and the response carries a warning suggesting Imagen for strict API-level enforcement (reverts the auto-routing added in 0.5.1)

## [0.5.2] - 2026-03-30

### Fixed

- Convert `google-genai` `Image` objects to Pillow before saving in the Imagen code paths

## [0.5.1] - 2026-03-30

### Fixed

- `person_generation="block"` now maps to the correct SDK enum member `DONT_ALLOW`
- `person_generation` is no longer passed to Gemini `ImageConfig`, which raised `ValueError`

## [0.5.0] - 2026-03-22

### Added

- `transform_image` tool — crop, resize, rotate, flip, blur, sharpen, grayscale, watermark
- Prompt template library with `list_templates` and `apply_template` tools (24 templates)
- Optional Vertex AI tools: `upscale_image` (x2/x4) and `advanced_edit` (inpainting, outpainting, background swap, style transfer)
- Generation history with `list_history` and `get_generation_details` tools
- `batch_generate` tool — up to 10 prompts in parallel
- Quality presets (`fast`, `balanced`, `quality`) for `generate_image`
- Thinking mode (`thinking_budget`) and Google Search grounding on analysis tools
- Imagen 4 models: `imagen-4.0-generate-001`, `imagen-4.0-fast-generate-001`, `imagen-4.0-ultra-generate-001`
- EXIF metadata embedding (prompt, model, aspect ratio) in generated images
- Optional `model` override on all analysis methods in the API client

### Changed

- Migrated `edit`, `analyze`, and `remove_background` from `gemini-imagen` to the `google-genai` SDK; `gemini-imagen` dependency removed
- Upgraded to FastMCP 3.x
- Multi-image generation runs in parallel via `asyncio.gather`
- Server identifier renamed from `gemini-imagen-mcp` to `pixelforge-mcp`

### Fixed

- Path traversal in `edit_image` and `remove_background` output filenames
- Multi-image generation reported only the last result; per-image success is now tracked
- Thread-unsafe model mutation in `edit()`
- Safety blocks and rate-limit, permission, and invalid-argument errors are reported with specific messages
- API keys are never written to saved config files

## [0.3.0] - 2026-03-21

### Added

- `image_size` (1K/2K/4K), `number_of_images` (1-4), `output_format` (png/jpeg/webp), `person_generation`, and `reference_images` parameters for `generate_image`
- `model` and `output_format` parameters for `edit_image`
- New tools: `optimize_prompt`, `extract_text`, `detect_objects`, `remove_background`, `compare_images`, `estimate_cost`

### Fixed

- Safety settings are converted from preset strings to SDK objects
- RGBA-to-JPEG conversion composites onto a white background

## [0.2.0] - 2026-02-28

### Added

- New model `gemini-3.1-flash-image-preview` (Nano Banana 2) — Pro-level quality at Flash speed
- 4 panoramic aspect ratios: `1:4`, `4:1`, `1:8`, `8:1`
- Per-model temperature metadata in `list_available_models()` for AI client discoverability

### Changed

- Temperature range expanded from 0.0-1.0 to 0.0-2.0 to match API capabilities

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
- Temperature control for creativity adjustment (0.0-2.0)
- Pydantic-based configuration with YAML file support
- Environment variable support (GOOGLE_API_KEY, GOOGLE_GENERATIVE_AI_API_KEY, GEMINI_API_KEY)
- 72 unit tests with full coverage
- PyPI package distribution via pipx

[Unreleased]: https://github.com/tehnolabs/pixelforge-mcp/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/tehnolabs/pixelforge-mcp/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/tehnolabs/pixelforge-mcp/compare/v0.5.4...v0.6.0
[0.5.4]: https://github.com/tehnolabs/pixelforge-mcp/compare/v0.5.3...v0.5.4
[0.5.3]: https://github.com/tehnolabs/pixelforge-mcp/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/tehnolabs/pixelforge-mcp/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/tehnolabs/pixelforge-mcp/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/tehnolabs/pixelforge-mcp/compare/v0.3.0...v0.5.0
[0.3.0]: https://github.com/tehnolabs/pixelforge-mcp/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/tehnolabs/pixelforge-mcp/compare/v0.1.5...v0.2.0
[0.1.5]: https://github.com/tehnolabs/pixelforge-mcp/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/tehnolabs/pixelforge-mcp/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/tehnolabs/pixelforge-mcp/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/tehnolabs/pixelforge-mcp/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/tehnolabs/pixelforge-mcp/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/tehnolabs/pixelforge-mcp/releases/tag/v0.1.0
