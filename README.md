[![PyPI version](https://img.shields.io/pypi/v/pixelforge-mcp)](https://pypi.org/project/pixelforge-mcp/)
[![Python versions](https://img.shields.io/pypi/pyversions/pixelforge-mcp)](https://pypi.org/project/pixelforge-mcp/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

# PixelForge MCP

<p align="center">
  <img src="https://raw.githubusercontent.com/Tehnolabs/pixelforge-mcp/main/docs/assets/hero.png" alt="PixelForge MCP" width="600">
</p>

An MCP server for AI-powered image generation, editing, and analysis using Google's Gemini models.

## Features

- Generate images from text prompts with per-request model switching
- Edit existing images using natural language instructions
- Analyze images with AI-powered descriptions
- 10 aspect ratios and temperature control for creative flexibility
- Async-first architecture with full Pydantic validation
- Self-documenting tools with built-in model selection guidance

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

Ask Claude to generate, edit, or analyze images — the tools are available automatically.

## Available Tools

### generate_image

Generate an image from a text prompt.

**Parameters:**
- `prompt` (required): Text description of the image
- `output_filename` (optional): Custom filename
- `aspect_ratio` (optional): Image dimensions (default: "1:1")
- `temperature` (optional): Creativity level 0.0-1.0 (default: 0.7)
- `model` (optional): Model to use (default: "gemini-2.5-flash-image")
- `safety_setting` (optional): Content safety filter — "preset:strict" (default) or "preset:relaxed"

**Example prompts:**
> Generate an image of a futuristic city at sunset with flying cars in 16:9 widescreen

> Create a watercolor painting of a cat sleeping on a bookshelf, use the pro model

> Generate a minimalist logo for a coffee shop called "Bean There" in square format with high creativity

### edit_image

Edit an existing image with a text prompt.

**Parameters:**
- `prompt` (required): Description of desired changes
- `input_image_path` (required): Path to the image to edit
- `output_filename` (optional): Custom filename for edited image
- `temperature` (optional): Creativity level 0.0-1.0 (default: 0.7)

**Example prompts:**
> Edit this image and add a rainbow in the sky

> Remove the background and replace it with a gradient

> Make this photo look like it was taken during golden hour

### analyze_image

Get an AI-powered description and analysis of an image.

**Parameters:**
- `image_path` (required): Path to the image to analyze
- `prompt` (optional): Custom analysis prompt — directs the AI to focus on specific aspects instead of giving a general description

**Example prompts:**
> Analyze this image and describe what you see

> What's in this screenshot?

> Extract all visible text from this image (OCR)

> Evaluate this image for web accessibility — describe alt text, contrast issues, and readability

> List the dominant colors and their approximate hex values in this design

### list_available_models

List all available Gemini image generation models with capabilities and selection guidance.

**Example prompts:**
> What image generation models are available?

> Which model should I use for photorealistic images?

### get_server_info

Get server configuration and status information.

**Example prompts:**
> Show me the PixelForge server configuration

> What's the default model and output directory?

## Model Selection & Switching

PixelForge supports **per-request model switching** — choose the right model for your task:

| Use case | Model | Why |
|----------|-------|-----|
| Fast iterations | `gemini-2.5-flash-image` (default) | Speed, lower cost |
| High quality output | `gemini-3-pro-image-preview` | Photorealism, complex scenes |
| Text in images | `gemini-3-pro-image-preview` | Legible text rendering |
| High resolution (2K/4K) | `gemini-3-pro-image-preview` | Native high-res support |

**Example prompts:**
> Generate a quick concept sketch of a logo

Uses default fast model (`gemini-2.5-flash-image`).

> Generate a photorealistic portrait with intricate details using the pro model in 16:9

Switches to quality model (`gemini-3-pro-image-preview`).

**Best Practices:**
1. Use `gemini-2.5-flash-image` (default) for rapid prototyping
2. Switch to `gemini-3-pro-image-preview` for production quality
3. Always use `gemini-3-pro-image-preview` for readable text in images
4. Ask Claude to *"list available models"* for detailed model metadata

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

## Documentation

- [Configuration Guide](docs/configuration.md) — detailed setup and environment options
- [Changelog](CHANGELOG.md) — version history

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code standards, and pull request guidelines.

## License

[AGPL-3.0](LICENSE)

## Acknowledgments

Built with [FastMCP](https://github.com/jlowin/fastmcp), [Pydantic](https://docs.pydantic.dev/), and [gemini-imagen](https://github.com/aviadr1/gemini-imagen).

## Author

Ahmed Al-Eryani @ Tehnolabs

---

**PixelForge MCP** - Forging pixels with AI
