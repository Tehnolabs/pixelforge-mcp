---
name: pixelforge-setup
description: Sets up and troubleshoots the PixelForge MCP server (Google Gemini image tools). Use it on first install and whenever the user asks to set up, configure or verify PixelForge, add a Google or Gemini API key, switch to a service account or Vertex AI, check which backend is active, or onboard PixelForge in Claude Code, Codex or Cursor. Also use it the moment any PixelForge tool fails with No credentials found, Invalid API key, Rate limit exceeded, Permission denied, Publisher model was not found, quota limit 0, a 429, 403 or 404, or the server will not start.
allowed-tools: mcp__plugin_pixelforge_pixelforge__get_server_info mcp__plugin_pixelforge_pixelforge__list_available_models mcp__plugin_pixelforge_pixelforge__estimate_cost mcp__plugin_pixelforge_pixelforge__optimize_prompt
---

# PixelForge setup and troubleshooting

PixelForge runs every tool on one of two Google backends: the Gemini API (an API key) or Vertex AI (a Google Cloud project with Application Default Credentials). The server reads its environment once at start, so every fix below ends with a new session.

## 1. Diagnose

1. Call the PixelForge `get_server_info` tool. Read `backend.name` (`gemini_api` or `vertex_ai`), `backend.project_id`, `backend.location`, `storage.output_directory` and `server.version`. If the call fails or the tool is missing, go to "Server not connected" in the error table.
2. Compare `backend.name` with what the user intends. `gemini_api` is also what an unconfigured server reports, so it does not prove a key exists. A service account that reports `gemini_api` means an API key variable exists somewhere and wins (see Gotchas).
3. Run a live check. `get_server_info` and `list_available_models` never contact Google, so they cannot prove the credentials work. Call the PixelForge `optimize_prompt` tool with a short prompt such as `a red bicycle`. It is a text-model call that costs a fraction of a cent and exercises credentials, backend and location end to end. A `success: false` result carries the decoded error; look it up in the table below.
4. Image quota is separate from text quota. When the user wants proof that image generation works, call `estimate_cost` with `operation="generate"` and `model="gemini-3.1-flash-lite-image"`, quote the price, and with consent call `generate_image` with `quality="fast"` and `number_of_images=1`. `generate_image` is deliberately left out of this skill's `allowed-tools`, so the permission prompt doubles as the consent gate for the paid call.
5. Report the backend, the project and location when on Vertex AI, the output directory, and the live-check result. Never repeat a key value, even partially.

## 2. Choose credentials

Default to a Gemini API key in `GOOGLE_API_KEY` (from aistudio.google.com/apikey). `GOOGLE_GENERATIVE_AI_API_KEY` and `GEMINI_API_KEY` are accepted too. A `limit: 0` rate-limit error means the project behind the key has no quota for that model; enable billing or request quota (see the error table), retrying cannot help.

Use a service account instead when the user already bills through a Google Cloud project or policy forbids API keys. Set `GOOGLE_CLOUD_PROJECT` to the project ID and `GOOGLE_APPLICATION_CREDENTIALS` to the absolute path of the key file (`/path/to/service-account.json`, outside any repository). The identity needs `roles/aiplatform.user` and the Vertex AI API enabled on the project. Leave `GOOGLE_CLOUD_LOCATION` unset; the default `global` is the only location that serves the models PixelForge uses. With `gcloud auth application-default login` or an attached cloud identity, set only `GOOGLE_CLOUD_PROJECT`.

Backend selection, as implemented:

- An API key variable selects `gemini_api`.
- `GOOGLE_CLOUD_PROJECT` with no API key selects `vertex_ai`.
- Both set: the key wins unless `GOOGLE_GENAI_USE_VERTEXAI` is `1`, `true`, `yes` or `on` (`GOOGLE_GENAI_USE_ENTERPRISE` is accepted as well).
- An empty, whitespace-only or unexpanded `${...}` value counts as unset, so a blank form field never shadows the other credential.

Read `references/credentials.md` when the user needs to create the service account with gcloud from scratch, or wants to register the server directly in a client instead of through the plugin.

## 3. Get the secrets to the server, per client

The plugin starts the server with `uvx pixelforge-mcp@latest`, so `uv` must be on the PATH of the process that launches the client. A plugin-managed server does not read a `.env` file or an `env` block the user edits by hand; each client has one supported route, and values go into the client, never into a file in the repository.

**Claude Code.** The `/plugin` install or enable flow shows three fields (Google API key, Google Cloud project ID, service account key file) and stores them in the credential store. Fill the key, or the project plus the key file, and leave the rest blank. A variable already exported in the shell that launched Claude Code wins over the plugin setting, and a blank setting arrives as an empty value, so a stale export in a shell profile explains a backend the form did not choose. A shell install (`claude plugin install`) does not prompt; run `claude plugin configure pixelforge@pixelforge` to see which options are unset, then enter the values in `/plugin` (or pipe a JSON object to `claude plugin configure pixelforge@pixelforge --values-stdin`). Prefer the `/plugin` route so the key never passes through an agent-run command. Users who earlier ran `claude mcp add pixelforge ...` now load the tools twice; have them run `claude mcp remove pixelforge`. Start a new session afterwards.

**Codex.** There is no install prompt. The plugin forwards every variable named in this skill (the three key names, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CLOUD_LOCATION`, `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_GENAI_USE_ENTERPRISE`) and nothing else, and only from the shell that launched Codex. Have the user `export` them in their shell profile, open a new terminal, and start a new `codex` session; tools and skills load at session start. A desktop launch that skips the shell profile sees none of them.

**Cursor.** The plugin's MCP config is read-only and holds placeholders for `GOOGLE_API_KEY`, `GOOGLE_CLOUD_PROJECT` and `GOOGLE_APPLICATION_CREDENTIALS`. Set the ones needed in the plugin's configuration form (Plugins, then Configure for this plugin); to use the service account, leave the key field empty. Whether the server also inherits the shell environment is undocumented, so do not rely on it. Reload the window afterwards. If no form is available, register the server directly in `.cursor/mcp.json` (see the reference).

## 4. Decode the error

The server retries 408, 429 and 5xx responses five times with backoff before reporting, so a message that reaches the user is final. Match on the prefix the server emits.

| Message contains | Cause | Fix |
|---|---|---|
| `No credentials found` | No API key variable and no `GOOGLE_CLOUD_PROJECT` reached the server process (blank fields count as absent) | Deliver the values through the client route above; new session |
| `Vertex AI backend selected but no project is configured` | `GOOGLE_GENAI_USE_VERTEXAI` is on without `GOOGLE_CLOUD_PROJECT` | Set the project, or turn the switch off |
| `Invalid API key` | Wrong, revoked or truncated key, or no key entered and the host passed the literal `${...}` placeholder through to the server | Enter a key, or a project for Vertex AI, through the client route; new session |
| `Rate limit exceeded after 5 attempts (RESOURCE_EXHAUSTED)` with `limit: 0` | The project has no quota for that model (unbilled key, fresh Vertex project) | Enable billing or request quota for the model; retrying cannot help |
| Same prefix without `limit: 0` | A burst of calls (`batch_generate`, `number_of_images` above 1) | Wait, then generate sequentially |
| `Permission denied (PERMISSION_DENIED)` with `has not been used in project` or `is disabled` | The API is not enabled on the project, or the identity lacks the role | Enable the Vertex AI API (or the Generative Language API for a key); grant `roles/aiplatform.user` |
| `Publisher model ... was not found` with a regional location in the path | `GOOGLE_CLOUD_LOCATION` is set to a region | Unset it (default `global`) |
| `Publisher model ... was not found` with `/locations/global/` | Model ID unknown or retired on Vertex AI, or the project cannot access it yet | Call `list_available_models` for current IDs; check project access |
| Any other model `not found` | Deprecated or mistyped model ID | Call `list_available_models`; prefer `quality` presets over a hard-coded model |
| Server not connected; the client reports a failed start naming `uvx` | `uv` is missing or not on the client's PATH, or the first start is still downloading the package | Install `uv` (docs.astral.sh/uv), make sure `uvx` is on PATH for the client process, reconnect; with no network the server cannot start. For a direct pipx install, `pixelforge-mcp` lives in `$HOME/.local/bin`; add that to PATH |

## Gotchas

- `get_server_info` reports configuration, not connectivity, and says `gemini_api` even when nothing is configured. Only a real call proves the credentials.
- Configuration is read once at server start. Changing a variable in the shell of a running session changes nothing; start a new session.
- Backend `gemini_api` while a service account was configured means a real key variable is set somewhere (shell profile, plugin form, an old `claude mcp add` entry). Unset it there, or set `GOOGLE_GENAI_USE_VERTEXAI=true` in the same environment.
- Never echo a key or print the service-account file. To check presence, run `test -n "$GOOGLE_API_KEY" && echo set`. Keep the key file outside the repository and gitignored, with only `roles/aiplatform.user`.
- `GOOGLE_APPLICATION_CREDENTIALS` must be an absolute path in JSON configs and plugin forms; `~` is not expanded there. An empty or unexpanded value is dropped so other Application Default Credentials sources still work.
- `uvx pixelforge-mcp@latest` resolves the newest release on every cold start, so a fresh session after a release downloads the package; give the first connection time before diagnosing credentials.
- Environment variables override `config/config.yaml`, which the server looks for relative to its own working directory.
- "Missing" images are in `storage.output_directory`, which defaults to `./generated_images` under the server's working directory, not necessarily the user's project.
- Both backends run every tool; nothing extra needs installing for Vertex AI.
- Plugin tool names carry a plugin prefix, so permission rules or hooks written for a standalone install stop matching after switching to the plugin.
