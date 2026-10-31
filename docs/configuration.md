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

## Plugin Install (Claude Code, Codex, Cursor)

The plugin bundles the server and seven workflow skills and starts the server with `uvx pixelforge-mcp@latest`, so it needs [uv](https://docs.astral.sh/uv/getting-started/installation/) rather than pipx.

| Client | Install | Credentials |
|--------|---------|-------------|
| Claude Code | `/plugin install pixelforge --marketplace Tehnolabs/pixelforge-mcp` | Prompted in `/plugin` (stored in the credential store) |
| Codex | `codex plugin marketplace add Tehnolabs/pixelforge-mcp` then `codex plugin add pixelforge@pixelforge` | Exported in the shell that launches Codex |
| Cursor | Customize → From GitHub Repository → `https://github.com/Tehnolabs/pixelforge-mcp` | The plugin's configuration form |

Details, including how to migrate from a `claude mcp add` setup, are in [plugins/pixelforge/README.md](../plugins/pixelforge/README.md). The sections below cover the manual, server-only setup.

## Client Configuration

The examples below use a Gemini API key. To use a Google Cloud service account instead, replace `GOOGLE_API_KEY` with `GOOGLE_CLOUD_PROJECT` and `GOOGLE_APPLICATION_CREDENTIALS` as shown in [Use a service account instead of an API key](#use-a-service-account-instead-of-an-api-key).

### Claude Code (Recommended)

```bash
claude mcp add pixelforge --scope user \
  -e GOOGLE_API_KEY="your-api-key-here" \
  -- pixelforge-mcp
```

This adds PixelForge to `~/.claude.json`, making it available in all your projects.

### Cursor

Add to `.cursor/mcp.json` in your project root (or global config):

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

### VS Code

```bash
code --add-mcp '{"name":"pixelforge","command":"pixelforge-mcp","env":{"GOOGLE_API_KEY":"your-api-key-here"}}'
```

This adds PixelForge to your VS Code MCP configuration.

### Windsurf

Add to `~/.codeium/windsurf/mcp_config.json`:

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
      "env": {
        "GOOGLE_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

Quit and restart Claude Desktop for changes to take effect.

### Other MCP Clients

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

### Vertex AI backend

Every tool can also run through Vertex AI. Use a Google Cloud project with the Vertex AI API enabled and provide [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials): a service account key file, `gcloud auth application-default login`, or the attached identity when running on Google Cloud. The identity needs the Vertex AI User role (`roles/aiplatform.user`). No extra packages are required.

| Variable | Default | Notes |
|----------|---------|-------|
| `GOOGLE_CLOUD_PROJECT` | — | Project ID. Setting it enables the Vertex AI backend |
| `GOOGLE_APPLICATION_CREDENTIALS` | — | Path to a service account key file. Not needed with another ADC source |
| `GOOGLE_CLOUD_LOCATION` | `global` | Vertex AI location. The current Gemini image and text models are served from the global endpoint |
| `GOOGLE_GENAI_USE_VERTEXAI` | `false` | `true` to use Vertex AI even when an API key is also set (`GOOGLE_GENAI_USE_ENTERPRISE` is accepted as well) |

### Use a service account instead of an API key

A service account runs every tool through Vertex AI, with no Gemini API key. One-time setup with the [gcloud CLI](https://cloud.google.com/sdk/docs/install), using your own project ID:

```bash
PROJECT_ID="YOUR_PROJECT_ID"
SA="pixelforge-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# 1. Enable Vertex AI in the project
gcloud services enable aiplatform.googleapis.com --project="$PROJECT_ID"

# 2. Create the service account and grant it the Vertex AI User role
gcloud iam service-accounts create pixelforge-sa --project="$PROJECT_ID"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SA" --role="roles/aiplatform.user"

# 3. Create a key file outside any repository
mkdir -p ~/.config/pixelforge
gcloud iam service-accounts keys create ~/.config/pixelforge/service-account.json \
  --iam-account="$SA"
```

Then replace the `GOOGLE_API_KEY` entry in your client configuration with `GOOGLE_CLOUD_PROJECT` and `GOOGLE_APPLICATION_CREDENTIALS`.

**Claude Code**

```bash
claude mcp add pixelforge --scope user \
  -e GOOGLE_CLOUD_PROJECT="YOUR_PROJECT_ID" \
  -e GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/pixelforge/service-account.json" \
  -- pixelforge-mcp
```

**Cursor, Windsurf, Claude Desktop and other JSON clients**

```json
{
  "mcpServers": {
    "pixelforge": {
      "command": "pixelforge-mcp",
      "env": {
        "GOOGLE_CLOUD_PROJECT": "YOUR_PROJECT_ID",
        "GOOGLE_APPLICATION_CREDENTIALS": "/absolute/path/to/service-account.json"
      }
    }
  }
}
```

**VS Code**

```bash
code --add-mcp '{"name":"pixelforge","command":"pixelforge-mcp","env":{"GOOGLE_CLOUD_PROJECT":"YOUR_PROJECT_ID","GOOGLE_APPLICATION_CREDENTIALS":"/absolute/path/to/service-account.json"}}'
```

**Kiro**

```bash
kiro-cli mcp add --name pixelforge --scope global --command pixelforge-mcp \
  --env "GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID,GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json"
```

**Verify**

Ask the client for `get_server_info`. `backend.name` is `vertex_ai` and `vertex_ai.location` is `global`. A `GOOGLE_CLOUD_LOCATION` entry is only needed for a non-default location.

- An API key variable set anywhere in the environment (a shell profile, for example) takes precedence, and the service account is ignored. Unset it, or add `"GOOGLE_GENAI_USE_VERTEXAI": "true"` to the same `env` block.
- The key file grants access to your project. Keep it out of version control, give the service account only `roles/aiplatform.user`, and delete keys you no longer use (`gcloud iam service-accounts keys list --iam-account="$SA"`).
- No key file is needed when credentials come from `gcloud auth application-default login` (optionally with `--impersonate-service-account="$SA"`) or from the attached identity on Google Cloud. Set only `GOOGLE_CLOUD_PROJECT` in that case.

### Backend selection

- An API key selects the Gemini API.
- `GOOGLE_CLOUD_PROJECT` with no API key selects Vertex AI.
- Both set: the API key wins unless `GOOGLE_GENAI_USE_VERTEXAI=true`.

`get_server_info` reports the active backend under `backend.name` (`gemini_api` or `vertex_ai`). The `pixelforge-mcp[vertex]` extra is kept for compatibility and installs nothing.

## Optional: YAML Configuration File

For advanced settings, create `config/config.yaml`:

```yaml
imagen:
  default_model: "gemini-nano-banana-2.1"
  default_aspect_ratio: "1:1"
  default_temperature: 0.7
  safety_setting: "preset:strict"

storage:
  output_dir: "./generated_images"

server:
  name: "pixelforge-mcp"
  version: "0.7.0"
  log_level: "INFO"

vertex:
  project_id: null      # or GOOGLE_CLOUD_PROJECT
  enabled: false        # true to use project_id above (set automatically by GOOGLE_CLOUD_PROJECT)
  location: "global"    # or GOOGLE_CLOUD_LOCATION
  prefer: false         # true to use Vertex AI even when an API key is set
```

In YAML, `project_id` selects Vertex AI only together with `enabled: true`; the `GOOGLE_CLOUD_PROJECT` variable sets both. Environment variables always override YAML values.

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
- Using a service account? An API key variable set anywhere in the environment takes precedence. Unset it or set `GOOGLE_GENAI_USE_VERTEXAI=true`

**"Publisher model ... was not found" (Vertex AI)**

- With a regional location in the message, the request went to a regional Vertex AI endpoint. The current Gemini image and text models are served from the global endpoint: set `GOOGLE_CLOUD_LOCATION=global` (the default) or remove a regional override
- With `locations/global` in the message, the model ID is unknown or retired on Vertex AI (`list_available_models` lists current IDs), or the project cannot access it yet
- The same 404 is returned for retired Imagen model IDs, which PixelForge no longer calls

**"Rate limit exceeded" or "Permission denied"**

- The message carries Google's own text. A quota `limit: 0` or "has not been used in project ... or it is disabled" is a project setup limit, not a model problem
- Enable the Vertex AI API (or the Generative Language API for an API key), grant `roles/aiplatform.user`, or request a quota increase

**"No credentials found"**

- Set an API key, or `GOOGLE_CLOUD_PROJECT` with Application Default Credentials (see [Vertex AI backend](#vertex-ai-backend))

**MCP server not connecting**

- Claude Code: Check `~/.claude.json` for correct configuration
- Cursor: Check `.cursor/mcp.json` for correct configuration
- VS Code: Verify MCP settings via `code --list-mcp`
- Windsurf: Check `~/.codeium/windsurf/mcp_config.json` for correct configuration
- Claude Desktop: Check `~/Library/Application Support/Claude/claude_desktop_config.json`
- Verify installation: `pipx list | grep pixelforge`

**Updating to a new version**

```bash
pipx upgrade pixelforge-mcp
```
