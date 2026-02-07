# PixelForge MCP - AI Agent Installation Guide

This guide is for AI agents (like Claude Code, Claude Desktop, and other MCP-compatible assistants) that need to install and configure PixelForge MCP.

## Quick Installation Summary

PixelForge MCP is a Model Context Protocol (MCP) server that provides AI-powered image generation, editing, and analysis using Google's Gemini models.

### Prerequisites

- Python 3.10 or higher
- Google API key ([Get one here](https://aistudio.google.com/apikey))
- pipx installed ([Installation guide](https://pipx.pypa.io/stable/installation/))

## Installation Methods

### Method 1: Global Installation with pipx (Recommended for AI Agents)

This installs PixelForge globally, making it available from any directory.

```bash
# From the project directory
pipx install --force .

# Or from PyPI (when published)
pipx install pixelforge-mcp
```

**Location after installation:** `~/.local/bin/pixelforge-mcp`

### Method 2: Development Installation

For development work or contributing:

```bash
# Clone the repository
git clone https://github.com/tehnolabs/pixelforge-mcp.git
cd pixelforge-mcp

# Create virtual environment and install
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

## Configuration

### 1. Set Google API Key

PixelForge supports multiple environment variable names:

```bash
# Any of these will work:
export GOOGLE_API_KEY="your-api-key-here"
export GOOGLE_GENERATIVE_AI_API_KEY="your-api-key-here"
export GEMINI_API_KEY="your-api-key-here"
```

### 2. Configure MCP Client

#### Claude Code (CLI)

Add to user-level config (available in all projects):

```bash
claude mcp add pixelforge --scope user \
  -e GOOGLE_API_KEY="your-api-key-here" \
  -- pixelforge-mcp
```

This adds PixelForge to `/Users/[username]/.claude.json` (user-level global config, available in all your projects).

#### Claude Desktop (GUI)

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

## Verification

### Test Installation

```bash
# Check if command is available
which pixelforge-mcp

# Should output: /Users/[username]/.local/bin/pixelforge-mcp
```

### Test MCP Server

The server will start when invoked by an MCP client. You should see:

```
Starting gemini-imagen-mcp v1.0.0
Output directory: generated_images
```

## Available Tools

Once installed, AI agents can use these tools:

1. **generate_image** - Generate images from text prompts
   - Supports model switching (gemini-2.5-flash-image, gemini-3-pro-image-preview)
   - 10 aspect ratios (1:1, 16:9, 9:16, etc.)
   - Temperature control (0.0-1.0)

2. **edit_image** - Modify existing images with text prompts

3. **analyze_image** - Get AI-powered image descriptions

4. **list_available_models** - Get detailed model capabilities and selection guidance

5. **get_server_info** - Get server configuration and model switching info

## Model Switching

PixelForge supports effortless model switching on every request:

```python
# Fast iterations
generate_image(
    prompt="Quick logo concept",
    model="gemini-2.5-flash-image"
)

# High quality final output
generate_image(
    prompt="Photorealistic portrait",
    model="gemini-3-pro-image-preview"
)
```

**Quick Selection Guide:**
- **Speed/iterations:** Use `gemini-2.5-flash-image` (default)
- **Quality/complexity:** Use `gemini-3-pro-image-preview`
- **Text in images:** Use `gemini-3-pro-image-preview`
- **High resolution (2K/4K):** Use `gemini-3-pro-image-preview`

Call `list_available_models()` for detailed capabilities and guidance.

## Reinstalling After Updates

When the PixelForge codebase is updated (e.g., after pulling new changes), reinstall to pick up improvements:

```bash
# Navigate to project directory
cd /path/to/pixelforge-mcp

# Reinstall with pipx (--force overwrites existing installation)
pipx install --force .
```

The `--force` flag ensures the latest code is deployed globally.

## Troubleshooting

### Issue: "Command not found: pixelforge-mcp"

**Solution:**
```bash
# Ensure pipx bin directory is in PATH
export PATH="$HOME/.local/bin:$PATH"

# Add to shell profile for persistence
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc  # or ~/.bashrc
```

### Issue: "Invalid API key" or "Authentication failed"

**Solution:**
- Verify your Google API key at [Google AI Studio](https://aistudio.google.com/apikey)
- Ensure the key has Gemini API access enabled
- Check that the environment variable is set correctly

### Issue: Changes not reflecting after update

**Solution:**
```bash
# Reinstall with --force
pipx install --force .

# Verify version
pipx list | grep pixelforge
```

### Issue: MCP server not starting

**Solution:**
```bash
# Check dependencies
pipx list

# Verify MCP client configuration
# For Claude Code: Check /Users/[username]/.claude.json
# For Claude Desktop: Check ~/Library/Application Support/Claude/claude_desktop_config.json
```

## Architecture

PixelForge follows a clean, modular architecture:

```
pixelforge-mcp/
├── src/pixelforge_mcp/
│   ├── config.py         # Pydantic-based configuration
│   ├── server.py         # FastMCP server with 5 async tools
│   └── utils/
│       ├── api_client.py # Direct API using gemini-imagen library
│       └── validation.py # Input validation with Pydantic
└── tests/
    └── unit/             # 72 unit tests (TDD approach)
```

## Key Features

- ✅ **Self-Contained:** No external CLI dependencies
- ✅ **Async-First:** All operations use async/await
- ✅ **Type-Safe:** Full Pydantic validation
- ✅ **Model Switching:** Change models per-request with zero overhead
- ✅ **AI-Friendly:** Rich metadata and guidance built into tool descriptions
- ✅ **Well-Tested:** 72 unit tests with TDD methodology

## Resources

- **Repository:** https://github.com/tehnolabs/pixelforge-mcp
- **Google AI Studio:** https://aistudio.google.com/apikey
- **FastMCP Documentation:** https://github.com/jlowin/fastmcp
- **MCP Protocol:** https://modelcontextprotocol.io

---

**For AI Agents:** This guide provides everything you need to install, configure, and use PixelForge MCP. The server is designed to be AI-friendly with self-documenting tools and intelligent model selection guidance.
