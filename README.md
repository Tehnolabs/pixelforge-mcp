[![PyPI version](https://img.shields.io/pypi/v/pixelforge-mcp)](https://pypi.org/project/pixelforge-mcp/)
[![Python versions](https://img.shields.io/pypi/pyversions/pixelforge-mcp)](https://pypi.org/project/pixelforge-mcp/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

# PixelForge MCP

<p align="center">
  <img src="https://raw.githubusercontent.com/Tehnolabs/pixelforge-mcp/main/docs/assets/hero.png" alt="PixelForge MCP" width="600">
</p>

An MCP server for AI-powered image generation, editing, analysis, and transformation using Google's Gemini and Imagen 4 models.

## Features

- **21 MCP tools** for image generation, editing, analysis, transformation, and more
- **6 models** including Gemini image models and Imagen 4 family
- **Quality presets** (fast/balanced/quality) for simplified model selection
- **Parallel multi-image generation** via asyncio.gather
- **Image transforms** — crop, resize, rotate, flip, blur, sharpen, grayscale, watermark
- **Prompt template library** — 24 curated templates across 10 categories
- **Generation history** — full audit trail with search and pagination
- **Batch processing** — generate up to 10 images in parallel
- **Optional Vertex AI** — upscaling (x2/x4) and advanced editing modes
- **Thinking mode & grounding** — extended reasoning and Google Search for analysis
- 14 aspect ratios (including panoramic) and temperature control
- Async-first architecture with full Pydantic validation
- EXIF metadata embedding (prompt, model, timestamp)

## Quick Start

**Requirements:** Python 3.10+ and a [Google API key](https://aistudio.google.com/apikey)

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

Ask Claude to generate, edit, or analyze images — all 21 tools are available automatically.

## Available Tools

### Generation

| Tool | Description |
|------|-------------|
| `generate_image` | Generate images from text (6 models, 14 aspect ratios, quality presets, parallel multi-image) |
| `edit_image` | Modify existing images with text prompts |
| `remove_background` | Remove image background (transparent or white) |
| `transform_image` | Crop, resize, rotate, flip, blur, sharpen, grayscale, or watermark |
| `batch_generate` | Generate up to 10 images in parallel from multiple prompts |

### Analysis

| Tool | Description |
|------|-------------|
| `analyze_image` | AI-powered image description with optional grounding |
| `extract_text` | OCR — extract text with confidence levels |
| `detect_objects` | Detect objects with bounding boxes |
| `compare_images` | Compare 2-10 images for differences |

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

### Vertex AI (Optional)

| Tool | Description |
|------|-------------|
| `upscale_image` | Upscale images x2 or x4 (requires Vertex AI) |
| `advanced_edit` | Inpainting, outpainting, background swap, style transfer (requires Vertex AI) |

## Model Selection

PixelForge supports **per-request model switching** with 6 models:

### Gemini Models (via Gemini API)

| Use case | Model | Why |
|----------|-------|-----|
| Fast iterations | `gemini-2.5-flash-image` (default) | Cheapest, lowest latency |
| Panoramic & grounded | `gemini-3.1-flash-image-preview` | 1:4/4:1/1:8/8:1, web+image grounding |
| Max text fidelity | `gemini-3-pro-image-preview` | ~94% accuracy, complex multi-turn edits |

### Imagen 4 Models (via Gemini API)

| Use case | Model | Why |
|----------|-------|-----|
| Cost-effective batch | `imagen-4.0-generate-001` | $0.04/img, excellent quality |
| Cheapest generation | `imagen-4.0-fast-generate-001` | $0.02/img, fastest |
| Maximum quality | `imagen-4.0-ultra-generate-001` | $0.06/img, best output |

### Quality Presets

Instead of choosing a model manually, use quality presets:

```
generate_image(prompt="...", quality="fast")      # gemini-2.5-flash-image
generate_image(prompt="...", quality="balanced")   # gemini-3.1-flash-image-preview + 1K
generate_image(prompt="...", quality="quality")    # gemini-3-pro-image-preview + 2K
```

## Vertex AI (Optional)

For advanced features like image upscaling and specialized editing modes:

1. Set up a Google Cloud project with Vertex AI enabled
2. Set the environment variable: `GOOGLE_CLOUD_PROJECT=your-project-id`
3. Install the Vertex AI dependency: `pip install pixelforge-mcp[vertex]`

PixelForge auto-detects Vertex AI credentials and unlocks `upscale_image` and `advanced_edit` tools.

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

\* Panoramic ratios require `gemini-3.1-flash-image-preview` model

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

### Vertex AI features not available

Ensure `GOOGLE_CLOUD_PROJECT` is set and `google-cloud-aiplatform` is installed:

```bash
pip install pixelforge-mcp[vertex]
```

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
