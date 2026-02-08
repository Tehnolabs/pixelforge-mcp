# Configuration Guide

## Installation

```bash
pipx install pixelforge-mcp
```

Verify the installation:

```bash
which pixelforge-mcp
# Expected: ~/.local/bin/pixelforge-mcp
```

## Client Configuration

### Option 1: Claude Code CLI (Recommended)

```bash
claude mcp add pixelforge --scope user \
  -e GOOGLE_API_KEY="your-api-key-here" \
  -- pixelforge-mcp
```

This adds PixelForge to `~/.claude.json`, making it available in all your projects.

### Option 2: Claude Desktop

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

### Option 3: Manual JSON Configuration

For other MCP-compatible clients, use this JSON structure:

```json
{
  "type": "stdio",
  "command": "pixelforge-mcp",
  "env": {
    "GOOGLE_API_KEY": "your-api-key-here"
  }
}
```

## Environment Variables

PixelForge checks these variables for your Google API key (in order):

| Variable | Notes |
|----------|-------|
| `GOOGLE_API_KEY` | Standard, recommended |
| `GOOGLE_GENERATIVE_AI_API_KEY` | Alternative |
| `GEMINI_API_KEY` | Alternative |

Get a key at [Google AI Studio](https://aistudio.google.com/apikey). Ensure the key has Gemini API access enabled.

## Optional: YAML Configuration File

For advanced settings, create `config/config.yaml`:

```yaml
imagen:
  default_model: "gemini-2.5-flash-image"
  default_aspect_ratio: "1:1"
  default_temperature: 0.7
  safety_setting: "preset:strict"

storage:
  output_dir: "./generated_images"

server:
  name: "pixelforge-mcp"
  version: "0.1.1"
  log_level: "INFO"
```

Environment variables always override YAML values.

## Troubleshooting

**"Command not found: pixelforge-mcp"**

Ensure the pipx bin directory is in your PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"

# Add to shell profile for persistence
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
```

**"Invalid API key" or "Authentication failed"**

- Verify your key at [Google AI Studio](https://aistudio.google.com/apikey)
- Ensure Gemini API access is enabled on the key
- Check the environment variable is set: `echo $GOOGLE_API_KEY`

**MCP server not connecting**

- Claude Code: Check `~/.claude.json` for correct configuration
- Claude Desktop: Check `~/Library/Application Support/Claude/claude_desktop_config.json`
- Verify installation: `pipx list | grep pixelforge`

**Updating to a new version**

```bash
pipx upgrade pixelforge-mcp
```
