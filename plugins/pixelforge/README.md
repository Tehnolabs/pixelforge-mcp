# PixelForge plugin

Image generation, editing, analysis and transformation for coding agents, backed by Google's Gemini image models. The plugin bundles the [pixelforge-mcp](https://github.com/Tehnolabs/pixelforge-mcp) server (19 tools) and seven skills that teach the agent proven workflows: README hero images, Open Graph cards, icon and favicon sets, diagrams and infographics, consistent characters, editing and upscaling, image analysis, and first-run setup.

The server starts with `uvx pixelforge-mcp@latest`, so each new session runs the current release from PyPI. The only prerequisite is [uv](https://docs.astral.sh/uv/getting-started/installation/) on your PATH.

## Install

### Claude Code

```bash
claude plugin marketplace add Tehnolabs/pixelforge-mcp
claude plugin install pixelforge@pixelforge
```

Then run `/plugin`, open PixelForge and enter either a Google API key or a Google Cloud project ID plus a service account key file. Values are stored in the platform credential store, not in settings files. A variable already exported in your shell (`GOOGLE_API_KEY`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_APPLICATION_CREDENTIALS`) takes precedence over the plugin setting. If you added the server earlier with `claude mcp add pixelforge`, remove that entry first (`claude mcp remove pixelforge`) so you do not load the tools twice.

### Codex

```bash
codex plugin marketplace add Tehnolabs/pixelforge-mcp
codex plugin add pixelforge@pixelforge
```

Codex forwards only the credential variables listed in the plugin, so export them in the shell before you launch Codex, then start a new session:

```bash
export GOOGLE_API_KEY="your-key"
# or: export GOOGLE_CLOUD_PROJECT="YOUR_PROJECT_ID" GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

### Cursor

Open **Customize**, choose **From GitHub Repository** and enter `https://github.com/Tehnolabs/pixelforge-mcp`, or paste the URL into `/add-plugin`. In the plugin's configuration form set `GOOGLE_API_KEY`, or `GOOGLE_CLOUD_PROJECT` and `GOOGLE_APPLICATION_CREDENTIALS` for a service account.

## Credentials

Any one of these works. An API key takes precedence over a service account unless `GOOGLE_GENAI_USE_VERTEXAI=true` is set.

| Variable | Purpose |
|----------|---------|
| `GOOGLE_API_KEY` | Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey) |
| `GOOGLE_CLOUD_PROJECT` | Google Cloud project ID; selects the Vertex AI backend |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to a service account JSON key with the Vertex AI User role |

The full service-account walkthrough is in the [repository README](https://github.com/Tehnolabs/pixelforge-mcp#use-a-service-account-instead-of-an-api-key). Ask the agent for `get_server_info` to confirm which backend is active.

## Skills

| Skill | Use it for |
|-------|-----------|
| `pixelforge-setup` | First run, credentials, backend precedence, decoding 403/404/429 errors |
| `pixelforge-generate` | New images from text: presets, ratios and sizes, cost gate, prompt structure, verification loop |
| `pixelforge-edit` | Editing existing images: transform, prompt edits, the six advanced modes, masks, upscaling |
| `pixelforge-analyze` | Descriptions, alt text, OCR, object detection, screenshot-versus-mockup comparison |
| `pixelforge-web-assets` | README hero images, Open Graph cards at 1200x630, icon and favicon families |
| `pixelforge-diagrams` | Diagrams, infographics and slide visuals with an OCR round-trip on every label |
| `pixelforge-characters` | Consistent characters, mascots and brand assets across many images |

Skills follow the [Agent Skills](https://agentskills.io) standard, so the same folder serves Claude Code, Codex and Cursor. In Claude Code they appear as `/pixelforge:<skill>`, in Codex as `$<skill>`, in Cursor as `/<skill>`.

## Layout

```
plugins/pixelforge/
├── .claude-plugin/plugin.json   # Claude Code manifest (credentials via userConfig)
├── .codex-plugin/plugin.json    # Codex manifest (credentials via env_vars passthrough)
├── .cursor-plugin/plugin.json   # Cursor manifest (credentials via plugin variables)
├── mcp/{claude,codex,cursor}.json
├── skills/<name>/SKILL.md       # shared by every client
└── assets/icon.svg
```

Each client expands environment placeholders differently, so every manifest points at its own MCP file. The skills and the server are shared.

## License

AGPL-3.0-or-later, the same as the server.
