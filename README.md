# PixelForge MCP

A production-ready MCP (Model Context Protocol) server for AI-powered image generation, editing, and analysis.

## About

PixelForge MCP provides a seamless bridge between Claude AI and Google's Gemini image generation API, enabling AI assistants to create, modify, and analyze images through natural language.

## Features

- 🎨 **Generate Images**: Create images from text descriptions with multi-model support
- ✏️ **Edit Images**: Modify existing images using text prompts
- 🔍 **Analyze Images**: Get AI-powered descriptions and insights
- 🎯 **10 Aspect Ratios**: Support for 1:1, 16:9, 9:16, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 21:9
- 🎚️ **Temperature Control**: Adjust creativity level (0.0-1.0)
- ✅ **Full Test Coverage**: 72 unit tests, built with TDD approach
- 🔒 **Type Safe**: Full Pydantic validation and type hints

## Prerequisites

- **Python 3.10+**
- **pipx** ([Installation guide](https://pipx.pypa.io/stable/installation/))
- **Google API key** ([Get one here](https://aistudio.google.com/apikey))

## Installation

### Global Installation with pipx (Recommended)

```bash
pipx install .

# Or from PyPI (once published)
pipx install pixelforge-mcp
```

This installs PixelForge globally in `~/.local/bin/`, making it accessible from any directory.

### Reinstalling After Updates

When the codebase is updated (e.g., after pulling new changes), reinstall to pick up improvements:

```bash
# Navigate to project directory
cd /path/to/pixelforge-mcp

# Reinstall with --force flag
pipx install --force .
```

The `--force` flag ensures the latest code is deployed globally, overwriting the previous installation.

### For Development

```bash
git clone https://github.com/tehnolabs/pixelforge-mcp.git
cd pixelforge-mcp

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### For AI Agents

**AI agents (Claude Code, Claude Desktop, etc.)** should use the global installation method above. See [CLAUDE.md](CLAUDE.md) for comprehensive AI agent installation and configuration guide, including:
- Step-by-step installation instructions
- MCP client configuration examples
- Model switching guidance
- Troubleshooting common issues
- Verification steps

## Configuration

1. **Get your Google API key** from [Google AI Studio](https://aistudio.google.com/apikey)

2. **Set the API key** as an environment variable (supports multiple formats):

```bash
# Any of these will work:
export GOOGLE_API_KEY="your-api-key-here"
export GOOGLE_GENERATIVE_AI_API_KEY="your-api-key-here"
export GEMINI_API_KEY="your-api-key-here"
```

3. **(Optional)** Create a config file at `config/config.yaml`:

```yaml
imagen:
  api_key: "your-api-key"
  default_model: "gemini-2.5-flash-image"
  default_aspect_ratio: "1:1"
  default_temperature: 0.7
  safety_setting: "preset:strict"

storage:
  output_dir: "./generated_images"

server:
  name: "pixelforge-mcp"
  version: "1.0.0"
  log_level: "INFO"
```

## Usage with Claude

PixelForge works with both Claude Code (CLI) and Claude Desktop (GUI app).

### Claude Code (CLI)

Add globally for all projects:

```bash
claude mcp add pixelforge --scope user -e GOOGLE_API_KEY="your-api-key-here" -- pixelforge-mcp
```

This adds to `/Users/[username]/.claude.json` (user-level global config, available in all your projects).

### Claude Desktop (GUI App)

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "pixelforge": {
      "command": "pixelforge-mcp",
      "args": [],
      "env": {
        "GOOGLE_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

**Important:** Completely quit and restart Claude Desktop for changes to take effect.

**Note:** After installing with pipx, the `pixelforge-mcp` command is globally available from `~/.local/bin/`.

## Available Tools

### generate_image

Generate an image from a text prompt.

**Parameters:**
- `prompt` (required): Text description of the image
- `output_filename` (optional): Custom filename
- `aspect_ratio` (optional): Image dimensions (default: "1:1")
- `temperature` (optional): Creativity level 0.0-1.0 (default: 0.7)
- `model` (optional): Specific model to use
- `safety_setting` (optional): Content safety filter (default: "preset:strict")

**Example:**
```python
generate_image(
    prompt="A futuristic city at sunset with flying cars",
    aspect_ratio="16:9",
    temperature=0.8
)
```

### edit_image

Edit an existing image with a text prompt.

**Parameters:**
- `prompt` (required): Description of desired changes
- `input_image_path` (required): Path to the image to edit
- `output_filename` (optional): Custom filename for edited image
- `temperature` (optional): Creativity level 0.0-1.0 (default: 0.7)

**Example:**
```python
edit_image(
    prompt="Add a rainbow in the sky",
    input_image_path="/path/to/image.png",
    temperature=0.7
)
```

### analyze_image

Get an AI-powered description and analysis of an image.

**Parameters:**
- `image_path` (required): Path to the image to analyze

**Example:**
```python
analyze_image(image_path="/path/to/image.png")
```

### list_available_models

List all available Gemini image generation models with detailed capabilities and metadata.

**Example:**
```python
list_available_models()
```

**Returns:**
```json
{
  "models": [
    {
      "name": "gemini-2.5-flash-image",
      "nickname": "Nano Banana",
      "speed": "fast",
      "quality": "good",
      "best_for": ["Quick iterations", "High-volume generation", ...],
      "capabilities": {...}
    },
    {
      "name": "gemini-3-pro-image-preview",
      "nickname": "Gemini 3 Pro Image",
      "speed": "moderate",
      "quality": "excellent",
      "best_for": ["Photorealistic outputs", "Complex scenes", ...],
      "capabilities": {...}
    }
  ],
  "recommendation": "Use gemini-2.5-flash-image for speed, gemini-3-pro-image-preview for quality"
}
```

### get_server_info

Get server configuration and status information including model switching capabilities.

**Example:**
```python
get_server_info()
```

## Model Selection & Switching

PixelForge supports **effortless model switching** on every API call. Choose the right model for your task:

### Available Models

#### 1. **gemini-2.5-flash-image** (Default)
- **Nickname:** Nano Banana
- **Speed:** Fast ⚡
- **Quality:** Good
- **Best For:**
  - Quick iterations and concept exploration
  - High-volume batch generation
  - Simple compositions and designs
  - When speed matters more than perfection

#### 2. **gemini-3-pro-image-preview**
- **Nickname:** Gemini 3 Pro Image
- **Speed:** Moderate
- **Quality:** Excellent ✨
- **Best For:**
  - Photorealistic final outputs
  - Complex multi-object scenes
  - Legible text rendering in images
  - Character consistency across images
  - Multi-turn image editing workflows
  - High-resolution outputs (2K/4K)

### How to Switch Models

**Per-Request Switching** (Recommended):

```python
# Use fast model for iterations
generate_image(
    prompt="Quick concept sketch of a logo",
    model="gemini-2.5-flash-image"
)

# Switch to quality model for final output
generate_image(
    prompt="Photorealistic portrait with intricate details",
    model="gemini-3-pro-image-preview",
    aspect_ratio="16:9"
)
```

**Model Selection Decision Tree:**

```
Need text in images? → gemini-3-pro-image-preview
Need high resolution (2K/4K)? → gemini-3-pro-image-preview
Need complex multi-object scene? → gemini-3-pro-image-preview
Need fast iterations? → gemini-2.5-flash-image (default)
Just exploring concepts? → gemini-2.5-flash-image (default)
```

### Best Practices

1. **Iteration Phase:** Use `gemini-2.5-flash-image` for rapid prototyping
2. **Final Output:** Switch to `gemini-3-pro-image-preview` for production quality
3. **Text Rendering:** Always use `gemini-3-pro-image-preview` for readable text
4. **Cost Optimization:** Use faster model for volume, quality model only when needed
5. **Check Capabilities:** Call `list_available_models()` to see detailed model metadata

## Development

### Run Unit Tests

```bash
pytest tests/unit/
```

### Run Integration Tests

```bash
export PATH="$PWD/venv/bin:$PATH"
python test_integration.py
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type check
mypy src/
```

## Requirements

- Python 3.10 or higher
- Google API key with Gemini API access
- gemini-imagen Python library (installed automatically with dependencies)

## Architecture

PixelForge follows a clean, modular architecture:

```
pixelforge-mcp/
├── src/pixelforge_mcp/
│   ├── config.py         # Pydantic-based configuration management
│   ├── server.py         # FastMCP server with 5 async tools
│   └── utils/
│       ├── api_client.py # Direct API client using gemini-imagen library
│       └── validation.py # Input validation with Pydantic
├── tests/
│   └── unit/             # 72 unit tests (TDD approach)
└── config/               # Configuration files
```

**Design Principles:**
- **Type Safety**: Full Pydantic validation and type hints throughout
- **Async-First**: All image operations use async/await for better performance
- **Direct API Integration**: Uses gemini-imagen Python library (no CLI subprocess calls)
- **Self-Contained**: No external dependencies beyond Python packages
- **Separation of Concerns**: Clean layer separation (config, API client, validation, server)
- **Test-Driven Development**: All code written with tests first
- **DRY & KISS**: Simple, maintainable code without duplication

## Supported Aspect Ratios

- `1:1` - Square (default)
- `16:9` - Widescreen landscape
- `9:16` - Mobile portrait
- `2:3` - Classic portrait
- `3:2` - Classic landscape
- `3:4` - Portrait
- `4:3` - Traditional landscape
- `4:5` - Instagram portrait
- `5:4` - Medium format
- `21:9` - Ultrawide

## Troubleshooting

### Issue: "Invalid API key" or "Authentication failed"

Double-check your Google API key is correct and has access to the Gemini API. Get a key at [Google AI Studio](https://aistudio.google.com/apikey).

### Issue: "Server not starting"

Check that all dependencies are installed:

```bash
pip list | grep -E "(fastmcp|gemini-imagen|pydantic)"
```

## License

AGPL-3.0 License - See LICENSE file for details

## Author

Ahmed Al-Eryani @ Tehnolabs

## Acknowledgments

Built using:
- [FastMCP](https://github.com/jlowin/fastmcp) - MCP server framework
- [Pydantic](https://docs.pydantic.dev/) - Data validation
- [gemini-imagen](https://github.com/aviadr1/gemini-imagen) - Google Gemini image generation Python library

---

**PixelForge MCP** - Forging pixels with AI
