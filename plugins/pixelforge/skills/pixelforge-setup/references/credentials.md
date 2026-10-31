# Credentials reference

Read this when the user needs to create a service account with gcloud, or wants to register the server in a client directly instead of through the plugin. Placeholders only: never write a real project ID, key or key-file contents into a repository file or a chat message.

## Create a service account with gcloud (one time)

Requires the gcloud CLI, a Google Cloud project with billing, and permission to create service accounts in it.

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

Then deliver two values through the client route in SKILL.md: `GOOGLE_CLOUD_PROJECT="YOUR_PROJECT_ID"` and `GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/service-account.json"`. Leave the API key field or variable empty. Do not set `GOOGLE_CLOUD_LOCATION`; the default `global` is correct. Verify with the PixelForge `get_server_info` tool: `backend.name` is `vertex_ai` and `backend.location` is `global`. If it still says `gemini_api`, a real API key variable is set somewhere and wins; unset it or set `GOOGLE_GENAI_USE_VERTEXAI=true` in the same environment.

No key file is needed when Application Default Credentials come from `gcloud auth application-default login` (optionally with `--impersonate-service-account="$SA"`) or from the attached identity on Google Cloud; set only `GOOGLE_CLOUD_PROJECT` in that case.

Housekeeping: `gcloud iam service-accounts keys list --iam-account="$SA"` shows the keys that exist; delete the ones no longer in use. The key file grants access to the whole project within the role, so keep it out of version control and give the account nothing beyond `roles/aiplatform.user`.

## Register the server directly (without the plugin)

Use this when the client cannot install the plugin, when the plugin's configuration form is not available, or when the user prefers a plain MCP entry. The launcher below matches the plugin (`uvx pixelforge-mcp@latest`, needs `uv` on PATH). A pipx install works the same way: `pipx install pixelforge-mcp`, then use `pixelforge-mcp` as the command with no arguments. For Vertex AI, replace the single `GOOGLE_API_KEY` entry with `GOOGLE_CLOUD_PROJECT` and `GOOGLE_APPLICATION_CREDENTIALS` (absolute path).

Claude Code:

```bash
claude mcp add pixelforge --scope user \
  -e GOOGLE_API_KEY="your-api-key-here" \
  -- uvx pixelforge-mcp@latest
```

Codex:

```bash
codex mcp add pixelforge --env GOOGLE_API_KEY=your-api-key-here -- uvx pixelforge-mcp@latest
```

Cursor (`.cursor/mcp.json` in the project, or `~/.cursor/mcp.json`). Cursor expands `${env:NAME}` in a user or project `mcp.json`, so the key can stay in the shell instead of the file:

```json
{
  "mcpServers": {
    "pixelforge": {
      "command": "uvx",
      "args": ["pixelforge-mcp@latest"],
      "env": { "GOOGLE_API_KEY": "${env:GOOGLE_API_KEY}" }
    }
  }
}
```

Service-account `env` block for any JSON-configured client:

```json
"env": {
  "GOOGLE_CLOUD_PROJECT": "YOUR_PROJECT_ID",
  "GOOGLE_APPLICATION_CREDENTIALS": "/absolute/path/to/service-account.json"
}
```

Other clients take the same `command`, `args` and `env` block. Config locations when the server does not connect: Claude Code `~/.claude.json`; Cursor `.cursor/mcp.json`; VS Code user MCP settings (open the Command Palette and run MCP: List Servers); Windsurf `~/.codeium/windsurf/mcp_config.json`; Claude Desktop `~/Library/Application Support/Claude/claude_desktop_config.json` (quit and restart the app). In every client, restart the session after editing so the server starts with the new environment.

## What `get_server_info` reports

- `backend.name`, `backend.project_id`, `backend.location`: the active backend and, on Vertex AI, the project and location in use. `gemini_api` is also the answer when no credential is configured at all.
- `vertex_ai.enabled` is true when `GOOGLE_CLOUD_PROJECT` is set; `vertex_ai.active` is true only when that backend is the one serving requests.
- `storage.output_directory`: where generated files land; defaults to `./generated_images` under the server's working directory.
- `server.version`: the running server version. With the `@latest` launcher it is the newest release at the last cold start.
- The API key itself is never included.
