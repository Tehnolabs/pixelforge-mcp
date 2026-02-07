# MCP Server Configuration Instructions

## Installation

### Step 1: Install Globally with pipx

```bash
cd /path/to/pixelforge-mcp
pipx install .
```

This installs PixelForge globally in `~/.local/bin/pixelforge-mcp`, making it accessible from any directory.

### Step 2: Configure Claude Code MCP

## Method 1: Using Claude Code CLI (Recommended)

Add to Claude Code MCP configuration:

```bash
claude mcp add-json pixelforge '{
  "type": "stdio",
  "command": "pixelforge-mcp",
  "env": {
    "GOOGLE_API_KEY": "your-api-key-here"
  }
}'
```

Simple, clean, and truly global! No paths needed.

## Method 2: Using Project Directory (Development Only)

For development, you can use the local venv:

```bash
claude mcp add-json pixelforge '{
  "type": "stdio",
  "command": "/path/to/pixelforge-mcp/venv/bin/python",
  "args": ["-m", "pixelforge_mcp.server"],
  "env": {
    "GOOGLE_API_KEY": "your-api-key-here",
    "PATH": "/path/to/pixelforge-mcp/venv/bin:/usr/local/bin:/usr/bin:/bin"
  }
}'
```

**Note:** This method requires the PATH environment variable to include venv/bin.

## Method 3: Manual Configuration

Edit your Claude Desktop config at:
`~/Library/Application Support/Claude/claude_desktop_config.json`

Add this configuration (after installing with pipx):

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

**Note:** The command is just `pixelforge-mcp` - no paths, no args, beautifully simple!

## Verify Installation

After configuration, restart Claude and try:

```
"Generate an image of a sunset over mountains"
```

Claude should automatically use the pixelforge MCP server to generate the image.

## Available Tools

Once configured, Claude will have access to:

1. **generate_image** - Create images from text
2. **edit_image** - Modify existing images
3. **analyze_image** - Get descriptions of images
4. **list_available_models** - List Gemini models
5. **get_server_info** - Get server configuration

## Troubleshooting

### Issue: "Invalid API key"

Double-check your Google API key is correct and has access to Gemini API.

### Issue: "Server not starting"

Check that pixelforge-mcp is installed globally:

```bash
which pixelforge-mcp
pipx list | grep pixelforge
```

If not installed, run:

```bash
cd /path/to/pixelforge-mcp
pipx install .
```

## Testing the Server

You can test the server directly without Claude from any directory:

```bash
# Test the global command
pixelforge-mcp --help

# Or run integration tests
cd /path/to/pixelforge-mcp
python -m pytest tests/integration/ -v
```

Since it's installed globally with pipx, you can run `pixelforge-mcp` from anywhere!
