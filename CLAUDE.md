# PixelForge MCP - Development Guide

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

### Cursor

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "pixelforge": {
      "command": "pixelforge-mcp",
      "env": { "GOOGLE_API_KEY": "your-api-key-here" }
    }
  }
}
```

### VS Code

```bash
code --add-mcp '{"name":"pixelforge","command":"pixelforge-mcp","env":{"GOOGLE_API_KEY":"your-api-key-here"}}'
```

### Windsurf

Add to `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "pixelforge": {
      "command": "pixelforge-mcp",
      "env": { "GOOGLE_API_KEY": "your-api-key-here" }
    }
  }
}
```

### Kiro

```bash
kiro-cli mcp add --name pixelforge --scope global --command pixelforge-mcp --env "GOOGLE_API_KEY=your-api-key-here"
```

### Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "pixelforge": {
      "command": "pixelforge-mcp",
      "env": { "GOOGLE_API_KEY": "your-api-key-here" }
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
| `generate_image` | Generate images from text (model switching, 14 aspect ratios, temperature control) |
| `edit_image` | Modify existing images with text prompts |
| `analyze_image` | Get AI-powered image descriptions (supports custom analysis prompts) |
| `list_available_models` | Model capabilities and selection guidance |
| `get_server_info` | Server configuration and status |

## Model Selection

- **Fast iterations:** `gemini-2.5-flash-image` (default) — cheapest, lowest latency
- **Panoramic/grounded:** `gemini-3.1-flash-image-preview` — panoramic ratios, web+image grounding, fast 4K
- **Max text fidelity:** `gemini-3-pro-image-preview` — ~94% text accuracy, complex multi-turn edits

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
- Cursor: `.cursor/mcp.json`
- VS Code: User MCP settings
- Windsurf: `~/.codeium/windsurf/mcp_config.json`
- Claude Desktop: `~/Library/Application Support/Claude/claude_desktop_config.json`

## Resources

- [PyPI](https://pypi.org/project/pixelforge-mcp/)
- [Repository](https://github.com/tehnolabs/pixelforge-mcp)
- [Google AI Studio](https://aistudio.google.com/apikey)
- [MCP Protocol](https://modelcontextprotocol.io)

---

## Architecture

```
src/pixelforge_mcp/
├── __init__.py          # Package init
├── config.py            # Pydantic config, YAML loading, VERSION string
├── server.py            # FastMCP server — 5 async tool definitions + helpers
└── utils/
    ├── api_client.py    # ImagenAPIClient — wraps gemini-imagen library
    ├── cli.py           # ImagenCLI — subprocess-based fallback (legacy)
    └── validation.py    # Pydantic input models with validation rules
```

**Request flow:** MCP client → `server.py` (tool handler) → `validation.py` (validates input) → `api_client.py` (calls Gemini API) → response back to client.

**Key files for most changes:**
- `validation.py` — input models and constraints (Pydantic)
- `api_client.py` — API interaction logic
- `server.py` — tool registration and response formatting
- `config.py` — server configuration and version

## Development Workflow

### Branch naming

- `feat/` — new features (e.g., `feat/custom-analysis-prompts`)
- `fix/` — bug fixes (e.g., `fix/pypi-hero-image`)
- `docs/` — documentation only (e.g., `docs/update-readme`)
- `chore/` — maintenance (e.g., `chore/v0.1.3-hero-image`)

### Commit conventions

Use conventional commit prefixes:

- `feat:` — new feature
- `fix:` — bug fix
- `chore:` — maintenance, version bumps
- `docs:` — documentation changes
- `test:` — adding or updating tests
- `refactor:` — code restructuring without behavior change

### PR process

1. Branch from `main`
2. Make changes, add tests
3. Run all checks (see Code Style and Testing below)
4. Open PR against `main` with a clear description

## Adding or Modifying a Tool

Every tool change touches 3 files in order:

1. **`validation.py`** — Add or update the Pydantic input model (e.g., `AnalyzeImageInput`). This defines parameters, types, defaults, and validation rules.
2. **`api_client.py`** — Add or update the corresponding method on `ImagenAPIClient` (e.g., `analyze_image()`). This handles the actual API call.
3. **`server.py`** — Add or update the `@mcp.tool()` handler. This wires validation → API client → response formatting.

Then add tests in `tests/unit/` covering the new validation rules and API client behavior.

## Version Bumping

Version lives in **two places** — both must be updated together:

- `pyproject.toml` → `version = "X.Y.Z"` (line 3)
- `src/pixelforge_mcp/config.py` → `version: str = Field("X.Y.Z", ...)` (line 55)

**When to bump:** Ask the user before bumping. After bumping, add a CHANGELOG entry under the new version.

## Releasing

Releases are done manually from the command line. The `python-publish.yml` workflow handles PyPI upload automatically when a GitHub Release is created.

```bash
# 1. Make sure you're on main with the version already bumped
git checkout main && git pull

# 2. Create a GitHub Release (this triggers PyPI publish)
gh release create v0.X.Y --generate-notes
```

That's it. The `--generate-notes` flag auto-generates a changelog from merged PRs since the last release.

**Do not** use automated release workflows with `GITHUB_TOKEN` — GitHub blocks workflow-created events from triggering other workflows, so the PyPI publish step would never fire.

## Code Style

Configured in `pyproject.toml`:

| Tool | Purpose | Command |
|------|---------|---------|
| [Black](https://black.readthedocs.io/) | Formatting (line-length 88) | `black src/ tests/` |
| [Ruff](https://docs.astral.sh/ruff/) | Linting (E/F/I/N/W rules) | `ruff check src/ tests/` |
| [MyPy](https://mypy.readthedocs.io/) | Type checking (strict mode) | `mypy src/` |

Run all three before opening a PR.

## Testing

```bash
# Unit tests (no API key needed)
pytest tests/unit/ -v

# Specific test file
pytest tests/unit/test_validation.py -v

# Integration tests (requires GOOGLE_API_KEY)
pytest tests/integration/ -v
```

Test files mirror source structure:
- `test_validation.py` — input validation rules
- `test_api_client.py` — API client behavior (mocked)
- `test_config.py` — configuration loading
- `test_cli.py` — CLI wrapper
