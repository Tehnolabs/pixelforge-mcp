# PixelForge MCP - AI Agent Guide

MCP server for AI-powered image generation, editing, and analysis using Google's Gemini models.

## Prerequisites

- Python 3.10+
- Google API key ([Get one here](https://aistudio.google.com/apikey))
- pipx ([Installation guide](https://pipx.pypa.io/stable/installation/))

## Install

```bash
pipx install pixelforge-mcp
```

## Configure

### Claude Code

```bash
claude mcp add pixelforge --scope user \
  -e GOOGLE_API_KEY="your-api-key-here" \
  -- pixelforge-mcp
```

### Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "pixelforge": {
      "command": "pixelforge-mcp",
      "env": {
        "GOOGLE_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

Quit and restart Claude Desktop for changes to take effect.

## API Key

Any of these environment variables will work:

| Variable | Notes |
|----------|-------|
| `GOOGLE_API_KEY` | Standard |
| `GOOGLE_GENERATIVE_AI_API_KEY` | Alternative |
| `GEMINI_API_KEY` | Alternative |

## Available Tools

| Tool | Description |
|------|-------------|
| `generate_image` | Generate images from text (model switching, 10 aspect ratios, temperature control) |
| `edit_image` | Modify existing images with text prompts |
| `analyze_image` | Get AI-powered image descriptions |
| `list_available_models` | Model capabilities and selection guidance |
| `get_server_info` | Server configuration and status |

## Model Selection

- **Speed/iterations:** `gemini-2.5-flash-image` (default)
- **Quality/complexity:** `gemini-3-pro-image-preview`
- **Text in images:** `gemini-3-pro-image-preview`
- **High resolution:** `gemini-3-pro-image-preview`

Call `list_available_models()` for detailed guidance.

## Updating

```bash
pipx upgrade pixelforge-mcp
```

## Troubleshooting

**"Command not found: pixelforge-mcp"** — Ensure pipx bin directory is in PATH:
```bash
export PATH="$HOME/.local/bin:$PATH"
```

**"Invalid API key"** — Verify your key at [Google AI Studio](https://aistudio.google.com/apikey) and ensure Gemini API access is enabled.

**MCP server not starting** — Check your client configuration:
- Claude Code: `~/.claude.json`
- Claude Desktop: `~/Library/Application Support/Claude/claude_desktop_config.json`

## Resources

- [PyPI](https://pypi.org/project/pixelforge-mcp/)
- [Repository](https://github.com/tehnolabs/pixelforge-mcp)
- [Google AI Studio](https://aistudio.google.com/apikey)
- [MCP Protocol](https://modelcontextprotocol.io)
