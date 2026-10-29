# PixelForge MCP - Development Guide

MCP server for AI-powered image generation, editing, analysis, and transformation using Google's Gemini image models.

## Prerequisites

- Python 3.10+
- Google API key ([Get one here](https://aistudio.google.com/apikey)) or a Google Cloud project with Vertex AI
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

## Authentication

### API key

Any of these environment variables will work:

| Variable | Notes |
|----------|-------|
| `GOOGLE_API_KEY` | Standard |
| `GOOGLE_GENERATIVE_AI_API_KEY` | Alternative |
| `GEMINI_API_KEY` | Alternative |

### Vertex AI backend

Every tool can run through Vertex AI instead of an API key. Credentials come from Application Default Credentials; no extra packages are needed (`pixelforge-mcp[vertex]` is kept for compatibility and installs nothing).

| Variable | Notes |
|----------|-------|
| `GOOGLE_CLOUD_PROJECT` | Project ID. Setting it enables the Vertex AI backend |
| `GOOGLE_APPLICATION_CREDENTIALS` | Service account key file. Not needed with another ADC source |
| `GOOGLE_CLOUD_LOCATION` | Default `global`. The current Gemini image and text models are served from the global endpoint |
| `GOOGLE_GENAI_USE_VERTEXAI` | `true` to prefer Vertex AI when an API key is also set (`GOOGLE_GENAI_USE_ENTERPRISE` is accepted as well) |

Selection: an API key selects the Gemini API; `GOOGLE_CLOUD_PROJECT` with no key selects Vertex AI; with both, the key wins unless `GOOGLE_GENAI_USE_VERTEXAI=true`. `get_server_info` reports the active backend under `backend.name`.

## Available Tools (19)

### Generation
| Tool | Description |
|------|-------------|
| `generate_image` | Generate images from text (4 models, quality presets, parallel multi-image, thinking mode) |
| `edit_image` | Modify existing images with text prompts |
| `remove_background` | Remove image background |
| `transform_image` | Crop, resize, rotate, flip, blur, sharpen, grayscale, watermark |
| `batch_generate` | Generate up to 10 images in parallel |
| `upscale_image` | Generative upscale to the next 1K/2K/4K tier (re-render with a Gemini image model) |
| `advanced_edit` | Inpainting, outpainting, background swap, style transfer, product shots (prompt-driven, optional mask guidance) |

### Analysis
| Tool | Description |
|------|-------------|
| `analyze_image` | AI-powered image descriptions (with optional grounding and `model` override) |
| `extract_text` | OCR — extract text with confidence levels |
| `detect_objects` | Detect objects with bounding boxes |
| `compare_images` | Compare 2-10 images |

Analysis tools and `optimize_prompt` accept a `model` override (default `gemini-3.5-flash-lite`; or `gemini-3.8-flash` for harder OCR or reasoning).

### Utility
| Tool | Description |
|------|-------------|
| `optimize_prompt` | Enhance prompts for better generation (14 styles) |
| `estimate_cost` | Calculate costs per model/operation |
| `list_templates` | Browse 24 curated prompt templates |
| `apply_template` | Fill template with subject for ready-to-use prompt |
| `list_available_models` | Model capabilities and selection guidance |
| `get_server_info` | Server configuration and status |

### History
| Tool | Description |
|------|-------------|
| `list_history` | Browse generation history with filtering |
| `get_generation_details` | Get details of a specific generation |

## Model Selection

### Image Models
- **Default:** `gemini-nano-banana-2.1` — 1K/2K/4K, web+image grounding, thinking levels, panoramic ratios, up to 14 references, $0.0336/img at 1K
- **Fastest/cheapest:** `gemini-3.1-flash-lite-image` — 1K only, no panoramic, no grounding, $0.0336/img
- **Previous-gen workhorse:** `gemini-3.1-flash-image` — 0.5K/1K/2K/4K, panoramic, grounding, $0.067/img at 1K
- **Max text fidelity:** `gemini-3-pro-image` — best text rendering, complex multi-turn edits, $0.134/img at 1K/2K
- **Deprecated:** `gemini-2.5-flash-image` — shutdown announced, use `gemini-3.1-flash-lite-image`

Panoramic ratios (1:4, 4:1, 1:8, 8:1) require `gemini-nano-banana-2.1` or `gemini-3.1-flash-image`. `0.5K` is `gemini-3.1-flash-image` only and is sent to the API as `512px`; `gemini-3.1-flash-lite-image` is 1K only. `generate_image` rejects a size the chosen model cannot output.

`upscale_image` and `advanced_edit` re-render with `gemini-nano-banana-2.1` unless `model` is set. `upscale_image` outputs the smallest of 1K/2K/4K that covers the source times the factor (capped at 4K) and needs 2K/4K output (`gemini-nano-banana-2.1`, `gemini-3.1-flash-image`, or `gemini-3-pro-image`), so `gemini-3.1-flash-lite-image` is rejected there. Both tools pin the nearest aspect ratio the model supports; models without panoramic output get a standard ratio and a warning.

### Analysis Models
- **Default:** `gemini-3.5-flash-lite` — $0.30 in / $2.50 out per 1M tokens
- **Strongest:** `gemini-3.8-flash` — $0.75 in / $3.75 out per 1M tokens

### Quality Presets
Use `quality="fast"` (`gemini-3.1-flash-lite-image`), `"balanced"` (`gemini-nano-banana-2.1`, 1K), or `"quality"` (`gemini-3-pro-image`, 2K) instead of choosing a model manually.

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

**"Publisher model ... was not found" (Vertex AI)** — With a regional location in the message, set `GOOGLE_CLOUD_LOCATION=global` (the default); the current Gemini image and text models are served from the global endpoint. With `locations/global` in the message, the model ID is unknown or retired on Vertex AI (`list_available_models` lists current IDs), or the project cannot access it yet.

**"Rate limit exceeded" or "Permission denied"** — The message carries Google's own text. A quota `limit: 0` or "has not been used in project ... or it is disabled" is a project setup limit: enable the Vertex AI API (or the Generative Language API for an API key), grant `roles/aiplatform.user`, or request a quota increase.

**"No credentials found"** — Set an API key, or `GOOGLE_CLOUD_PROJECT` with Application Default Credentials.

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
├── config.py            # Pydantic config, YAML loading, VERSION, VertexConfig
├── server.py            # FastMCP server — 19 async tool definitions + helpers
├── templates/
│   ├── __init__.py      # Templates package
│   └── prompts.yaml     # 24 curated prompt templates
└── utils/
    ├── api_client.py    # ImagenAPIClient — google-genai SDK (Gemini API or Vertex AI)
    ├── history.py       # GenerationHistory — JSON append-only log
    ├── templates.py     # TemplateLibrary — load, list, apply templates
    ├── transforms.py    # Image transforms — crop, resize, rotate, etc.
    └── validation.py    # Pydantic input models with validation rules
```

**Request flow:** MCP client → `server.py` (tool handler) → `validation.py` (validates input) → `api_client.py` (calls the Gemini API or Vertex AI) → response back to client.

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

# Integration tests (requires GOOGLE_API_KEY, or GOOGLE_CLOUD_PROJECT with ADC)
pytest tests/integration/ -v
```

Test files mirror source structure:
- `test_validation.py` — input validation rules (110+ tests)
- `test_api_client.py` — API client behavior with mocked google-genai SDK
- `test_config.py` — configuration loading
- `test_server.py` — server handler tests (format_image_result, generate_output_path, estimate_cost)
- `test_transforms.py` — image transform operations
