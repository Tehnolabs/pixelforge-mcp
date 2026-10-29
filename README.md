[![PyPI version](https://img.shields.io/pypi/v/pixelforge-mcp)](https://pypi.org/project/pixelforge-mcp/)
[![Python versions](https://img.shields.io/pypi/pyversions/pixelforge-mcp)](https://pypi.org/project/pixelforge-mcp/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

# PixelForge MCP

<p align="center">
  <img src="https://raw.githubusercontent.com/Tehnolabs/pixelforge-mcp/main/docs/assets/hero.png" alt="PixelForge MCP" width="600">
</p>

An MCP server for AI-powered image generation, editing, analysis, and transformation using Google's Gemini image models.

## Features

- **19 MCP tools** for image generation, editing, analysis, transformation, and more
- **4 stable Gemini image models** (Nano Banana 2.1, 2 Lite, 2, and Pro) plus configurable analysis models
- **Quality presets** (fast/balanced/quality) for simplified model selection
- **Parallel multi-image generation** via asyncio.gather
- **Image transforms** — crop, resize, rotate, flip, blur, sharpen, grayscale, watermark
- **Prompt template library** — 24 curated templates across 10 categories
- **Generation history** — full audit trail with search and pagination
- **Batch processing** — generate up to 10 images in parallel
- **Generative upscale and mode-driven editing** — upscale to the next 1K/2K/4K tier, inpaint, outpaint, background swap, style transfer, product shots
- **Two backends** — Gemini API key, or Vertex AI with a service account / Application Default Credentials
- **Thinking mode & grounding** — extended reasoning and Google Search for analysis
- 14 aspect ratios (including panoramic) and temperature control
- Async-first architecture with full Pydantic validation
- EXIF metadata embedding (prompt, model, timestamp)

## Quick Start

**Requirements:** Python 3.10+ and a [Google API key](https://aistudio.google.com/apikey) or a Google Cloud project with Vertex AI (see [Authentication](#authentication))

### Install

```bash
pipx install pixelforge-mcp
```

### Configure

#### Claude Code

```bash
claude mcp add pixelforge --scope user -e GOOGLE_API_KEY="your-key" -- pixelforge-mcp
```

#### Cursor

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "pixelforge": {
      "command": "pixelforge-mcp",
      "env": {
        "GOOGLE_API_KEY": "your-key"
      }
    }
  }
}
```

#### VS Code

```bash
code --add-mcp '{"name":"pixelforge","command":"pixelforge-mcp","env":{"GOOGLE_API_KEY":"your-key"}}'
```

#### Windsurf

Add to `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "pixelforge": {
      "command": "pixelforge-mcp",
      "env": {
        "GOOGLE_API_KEY": "your-key"
      }
    }
  }
}
```

#### Kiro

```bash
kiro-cli mcp add --name pixelforge --scope global --command pixelforge-mcp --env "GOOGLE_API_KEY=your-key"
```

#### Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "pixelforge": {
      "command": "pixelforge-mcp",
      "env": {
        "GOOGLE_API_KEY": "your-key"
      }
    }
  }
}
```

Restart Claude Desktop after saving.

### Use

Ask Claude to generate, edit, or analyze images — all 19 tools are available automatically.

## Authentication

PixelForge talks to Google through one of two backends. Every tool runs on either.

### Gemini API key

Set any one of these variables. Get a key at [Google AI Studio](https://aistudio.google.com/apikey).

| Variable | Notes |
|----------|-------|
| `GOOGLE_API_KEY` | Standard |
| `GOOGLE_GENERATIVE_AI_API_KEY` | Alternative |
| `GEMINI_API_KEY` | Alternative |

### Vertex AI

Use a Google Cloud project with the Vertex AI API enabled and [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials): a service account key file, `gcloud auth application-default login`, or the attached identity when running on Google Cloud. The identity needs the Vertex AI User role (`roles/aiplatform.user`). No extra packages are required.

| Variable | Notes |
|----------|-------|
| `GOOGLE_CLOUD_PROJECT` | Project ID. Setting it enables the Vertex AI backend |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to a service account key file. Not needed with another ADC source |
| `GOOGLE_CLOUD_LOCATION` | Vertex AI location. Default `global`; the current Gemini image and text models are served from the global endpoint |
| `GOOGLE_GENAI_USE_VERTEXAI` | `true` to use Vertex AI even when an API key is also set (`GOOGLE_GENAI_USE_ENTERPRISE` is accepted as well) |

```json
{
  "mcpServers": {
    "pixelforge": {
      "command": "pixelforge-mcp",
      "env": {
        "GOOGLE_CLOUD_PROJECT": "my-project",
        "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/service-account.json"
      }
    }
  }
}
```

### Backend selection

- An API key selects the Gemini API.
- `GOOGLE_CLOUD_PROJECT` with no API key selects Vertex AI.
- Both set: the API key wins unless `GOOGLE_GENAI_USE_VERTEXAI=true`.

`get_server_info` reports the active backend under `backend.name`.

## Available Tools

### Generation

| Tool | Description |
|------|-------------|
| `generate_image` | Generate images from text (4 models, 14 aspect ratios, quality presets, parallel multi-image) |
| `edit_image` | Modify existing images with text prompts |
| `remove_background` | Remove image background (transparent or white) |
| `transform_image` | Crop, resize, rotate, flip, blur, sharpen, grayscale, or watermark |
| `batch_generate` | Generate up to 10 images in parallel from multiple prompts |
| `upscale_image` | Generative upscale to the next 1K/2K/4K tier (re-render with a Gemini image model) |
| `advanced_edit` | Inpainting, outpainting, background swap, style transfer, product shots (prompt-driven, optional mask guidance) |

### Analysis

| Tool | Description |
|------|-------------|
| `analyze_image` | AI-powered image description with optional grounding and `model` override |
| `extract_text` | OCR — extract text with confidence levels |
| `detect_objects` | Detect objects with bounding boxes |
| `compare_images` | Compare 2-10 images for differences |

Analysis tools and `optimize_prompt` accept a `model` override. The default is `gemini-3.5-flash-lite`; use `gemini-3.8-flash` for harder OCR or reasoning.

### Utility

| Tool | Description |
|------|-------------|
| `optimize_prompt` | Enhance prompts for better image generation (14 styles) |
| `estimate_cost` | Calculate generation costs per model/operation |
| `list_templates` | Browse 24 curated prompt templates across 10 categories |
| `apply_template` | Fill a template with your subject for a ready-to-use prompt |
| `list_available_models` | Model capabilities, speed, quality, and selection guidance |
| `get_server_info` | Server configuration and status |

### History

| Tool | Description |
|------|-------------|
| `list_history` | Browse generation history with pagination and filtering |
| `get_generation_details` | Get full details of a specific generation |

## Model Selection

PixelForge supports **per-request model switching** with 4 stable Gemini image models:

### Image Models

| Use case | Model | Why |
|----------|-------|-----|
| General use (default) | `gemini-nano-banana-2.1` | 1K/2K/4K, web+image grounding, thinking levels, 1:4/4:1/1:8/8:1, up to 14 references. $0.0336/img at 1K |
| Fastest, cheapest | `gemini-3.1-flash-lite-image` | 1K only, no panoramic, no grounding. $0.0336/img |
| Previous-gen workhorse | `gemini-3.1-flash-image` | 0.5K/1K/2K/4K, panoramic, grounding. $0.067/img at 1K |
| Max text fidelity | `gemini-3-pro-image` | Best text rendering, complex multi-turn edits. $0.134/img at 1K/2K |

`gemini-2.5-flash-image` is deprecated (Google has announced its shutdown) and remains only for existing setups. Use `gemini-3.1-flash-lite-image` instead.

`0.5K` is `gemini-3.1-flash-image` only and is sent to the API as `512px`. `generate_image` rejects a size the chosen model cannot output.

`upscale_image` and `advanced_edit` re-render with `gemini-nano-banana-2.1` unless `model` is set. `upscale_image` outputs the smallest of 1K/2K/4K that covers the source times the factor (capped at 4K) and needs a model with 2K/4K output (`gemini-nano-banana-2.1`, `gemini-3.1-flash-image`, or `gemini-3-pro-image`); `gemini-3.1-flash-lite-image` is rejected. Both tools pin the nearest aspect ratio the model supports; models without panoramic output get a standard ratio and a warning.

### Analysis Models (text)

Used by `analyze_image`, `extract_text`, `detect_objects`, `compare_images`, and `optimize_prompt`. Override per request with `model=`.

| Model | Notes |
|-------|-------|
| `gemini-3.5-flash-lite` (default) | $0.30 in / $2.50 out per 1M tokens |
| `gemini-3.8-flash` | Strongest. $0.75 in / $3.75 out per 1M tokens |

### Quality Presets

Instead of choosing a model manually, use quality presets:

```
generate_image(prompt="...", quality="fast")      # gemini-3.1-flash-lite-image
generate_image(prompt="...", quality="balanced")   # gemini-nano-banana-2.1 + 1K
generate_image(prompt="...", quality="quality")    # gemini-3-pro-image + 2K
```

## Supported Aspect Ratios

| Ratio | Description |
|-------|-------------|
| `1:1` | Square (default) |
| `16:9` | Widescreen landscape |
| `9:16` | Mobile portrait |
| `2:3` | Classic portrait |
| `3:2` | Classic landscape |
| `3:4` | Portrait |
| `4:3` | Traditional landscape |
| `4:5` | Instagram portrait |
| `5:4` | Medium format |
| `21:9` | Ultrawide |
| `1:4` | Tall panoramic* |
| `4:1` | Wide panoramic* |
| `1:8` | Extra tall panoramic* |
| `8:1` | Extra wide panoramic* |

\* Panoramic ratios require `gemini-nano-banana-2.1` or `gemini-3.1-flash-image`

## Troubleshooting

### "Invalid API key" or "Authentication failed"

Double-check your Google API key is correct and has access to the Gemini API. Get a key at [Google AI Studio](https://aistudio.google.com/apikey).

### "Command not found: pixelforge-mcp"

Ensure the pipx bin directory is in your PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### "Server not starting"

Check that pixelforge-mcp is installed:

```bash
pipx list | grep pixelforge
```

### "Publisher model ... was not found" (Vertex AI)

With a regional location in the message, the request went to a regional Vertex AI endpoint. The current Gemini image and text models are served from the global endpoint: set `GOOGLE_CLOUD_LOCATION=global` (the default) or remove a regional override. With `locations/global` in the message, the model ID is unknown or retired on Vertex AI (`list_available_models` lists current IDs), or the project cannot access it yet. The same 404 is returned for retired Imagen model IDs, which PixelForge no longer calls.

### "Rate limit exceeded" or "Permission denied"

The message carries Google's own text. A quota `limit: 0`, or "has not been used in project ... or it is disabled", is a project setup limit rather than a model problem: enable the Vertex AI API (or the Generative Language API for an API key), grant `roles/aiplatform.user`, or request a quota increase.

### "No credentials found"

Set an API key, or `GOOGLE_CLOUD_PROJECT` with Application Default Credentials. See [Authentication](#authentication).

## Documentation

- [Configuration Guide](docs/configuration.md) — detailed setup and environment options
- [Changelog](CHANGELOG.md) — version history

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code standards, and pull request guidelines.

## License

[AGPL-3.0](LICENSE)

## Acknowledgments

Built with [FastMCP](https://github.com/jlowin/fastmcp), [Pydantic](https://docs.pydantic.dev/), and [google-genai](https://github.com/googleapis/python-genai).

## Author

Ahmed Al-Eryani @ Tehnolabs

---

**PixelForge MCP** - Forging pixels with AI
