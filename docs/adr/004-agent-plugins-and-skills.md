# ADR-004: Agent Plugins and Skills for Claude Code, Codex and Cursor

**Date:** 2026-10-31
**Status:** Accepted
**Deciders:** Project maintainers

## Context

PixelForge is installed by hand: `pipx install pixelforge-mcp`, then a client-specific `claude mcp add`, `.cursor/mcp.json` or `config.toml` entry with an API key. Three things were missing:

1. **Discovery and one-command install.** Claude Code, Codex and Cursor each have a plugin marketplace model where a Git repository is the marketplace. A user who knows the repository name can install the server and its guidance in one step.
2. **Workflow knowledge.** The 19 tool descriptions explain parameters, but not the workflows that the research on how developers use image generation inside coding agents keeps asking for: README hero images, Open Graph cards at exact pixel sizes, icon and favicon families, diagrams whose labels survive rendering, characters that stay consistent across a series, and the no-alpha, text-rendering and cost gotchas that produce wasted calls. Skills (the Agent Skills open standard, `SKILL.md`) are the portable vehicle for that knowledge across all three clients.
3. **Credential handling.** `claude mcp add -e` writes the key as plain text into `~/.claude.json`. Plugin hosts offer credential stores or prompts.

Research on 2026-10-30 against primary sources (client docs and, for Codex and Cursor, the client source code and published JSON schemas) established the constraints below.

### Constraints that shaped the layout

| Client | Manifest | MCP file | How a secret reaches the server | Marketplace file |
|--------|----------|----------|----------------------------------|------------------|
| Claude Code | `.claude-plugin/plugin.json` | path given by `mcpServers` | `userConfig` option with `sensitive: true`, substituted as `${user_config.<key>}` into `env`; prompted in `/plugin`, stored in the platform credential store | `.claude-plugin/marketplace.json` |
| Codex | `.codex-plugin/plugin.json` (no `$schema`) | path given by `mcpServers` | Codex starts stdio servers with a minimal environment (`HOME`, `PATH`, `TMPDIR` and a few more). Only names listed in the server's `env_vars` are forwarded from the user's shell. A static `env` block would be a committed secret | `.agents/plugins/marketplace.json` (read before `.claude-plugin/marketplace.json`) |
| Cursor | `.cursor-plugin/plugin.json` (no `$schema`; the published schema rejects unknown keys) | path given by `mcpServers`, otherwise `mcp.json` | `variables` (a JSON Schema object) filled in the plugin's configuration form, substituted as `${VAR}` into `env` | `.cursor-plugin/marketplace.json` (`source` must be a string path) |

Two facts rule out the obvious "one portable manifest":

- A root `plugin.json` carrying the Agent Plugins 1.0 `$schema` makes Codex treat the package as portable and ignore the Codex overlay entirely, including its `mcpServers`. The portable `mcp.json` schema has no `env_vars`, so a portable stdio server never receives the API key from the shell.
- The same filename means different things: Claude Code's `.mcp.json` expands `${user_config.x}`, Codex's `.mcp.json` wants `env_vars`, Cursor's `mcp.json` wants `${VAR}` plugin variables. A shared file would be wrong for two of the three.

A plugin at the repository root fails `claude plugin validate --strict` because a `CLAUDE.md` at the plugin root is not loaded as context (the repository's `CLAUDE.md` symlink only passes because the validator does not follow symlinks), and it would copy `src/`, `tests/` and `docs/` into every user's plugin cache.

The repository is AGPL-3.0. Cursor's marketplace publisher terms accept permissive licenses only, so an official Cursor listing is out of scope; installing from the GitHub repository works regardless.

## Decision

### 1. One bundle in `plugins/pixelforge/`, three thin manifests, one skills folder

```
plugins/pixelforge/
├── .claude-plugin/plugin.json   # userConfig for the key, project ID and key file path
├── .codex-plugin/plugin.json    # interface block, onboardingSkill -> pixelforge-setup
├── .cursor-plugin/plugin.json   # variables for the same three settings
├── mcp/claude.json              # env: ${user_config.*}
├── mcp/codex.json               # env_vars: passthrough list, startup_timeout_sec 90
├── mcp/cursor.json              # env: ${GOOGLE_API_KEY} ... (plugin variables)
├── skills/<name>/SKILL.md       # shared; references/ one level deep
├── assets/icon.svg
└── README.md
```

Each manifest points `mcpServers` at its own file under `mcp/`, so no `.mcp.json`, `mcp.json` or root `plugin.json` exists in the plugin directory. Codex discovers `.codex-plugin` before `.claude-plugin` and `.cursor-plugin`, Cursor reads only its own, Claude Code reads only its own; the three coexist.

Marketplace files live at the repository root and all point at `./plugins/pixelforge`. The marketplace and the plugin are both named `pixelforge`, so the install identifiers are `pixelforge@pixelforge` in Claude Code and Codex.

### 2. Launcher: `uvx pixelforge-mcp@latest`

The plugin starts the server with `uvx pixelforge-mcp@latest`. `uv` resolves the newest release from PyPI, caches the environment, and reuses it; nothing has to be installed beforehand and nothing in the plugin has to change when the server is released. The maintainers rejected a pinned `==X.Y.Z` because every release would require a plugin edit, and a bare `pixelforge-mcp` because it needs a prior `pipx install`.

Trade-offs accepted: `uv` must be on the user's PATH (one documented prerequisite); the first start downloads the package and its dependencies, so the Codex entry raises `startup_timeout_sec` to 90; `@latest` consults the index when it starts, so a fully offline machine runs the cached version only if the resolution falls back to it. The manual `pipx` route in the README is unchanged for users who want a fixed version.

### 3. Credentials per client, one server-side guard

All three manifests expose the same three optional settings: `GOOGLE_API_KEY`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_APPLICATION_CREDENTIALS`. A user supplies either the key or the project plus key file. The precedence rule from ADR-003 is unchanged (the key wins unless `GOOGLE_GENAI_USE_VERTEXAI=true`).

The Claude Code mapping is `${GOOGLE_API_KEY:-${user_config.google_api_key}}` (and likewise for the other two): a variable exported in the user's shell wins, otherwise the plugin setting, otherwise an empty string. Probed on 2026-10-31 with a plugin whose command dumped its environment: the plain `${user_config.x}` form replaces a shell export with an empty string, and the reverse nesting `${user_config.x:-${VAR}}` makes Claude Code drop the server. Only the shell-first form behaves in all three states.

Plugin hosts differ in what they substitute for an option the user left empty: an empty string, or the literal placeholder. `Config.load` now treats an empty value, or a value of the form `${...}`, as unset (`env_value()` in `config.py`), and removes such a `GOOGLE_APPLICATION_CREDENTIALS` from the environment so `google-auth` does not fail on a missing file. Without this, a literal `${user_config.google_api_key}` would be accepted as an API key and shadow a configured service account.

### 4. Skills hold workflows, the server holds facts

Seven skills ship: `pixelforge-setup`, `pixelforge-generate`, `pixelforge-edit`, `pixelforge-analyze`, `pixelforge-web-assets`, `pixelforge-diagrams`, `pixelforge-characters`. Rules:

- Frontmatter is limited to the Agent Skills fields plus `allowed-tools` (`name`, `description`, `allowed-tools`); `name` equals the directory name, so one folder serves all three clients.
- The body teaches procedure, defaults and gotchas; it does not repeat parameter lists from the tool descriptions. Prices and model lists are fetched at runtime through `list_available_models` and `estimate_cost`, so a model change does not stale the skills. Structural facts enforced by `validation.py` (size tiers, panoramic ratios and the models that support them, reference-image caps) may be stated.
- `SKILL.md` stays under 200 lines; long tables go into `references/` one level deep, with a sentence saying when to read each file.
- Tools are named by their bare name in prose ("the PixelForge `generate_image` tool"); only `allowed-tools` uses the Claude Code plugin form `mcp__plugin_pixelforge_pixelforge__<tool>`.

### 5. Server `instructions` for every client

`FastMCP("pixelforge-mcp", instructions=...)` sends a routing summary (tool selection, size and ratio rules, cost and credential hints) in the initialize response. Clients that do not load plugins, or users who install the server by hand, get the same rules. The text is kept under 4,096 characters (the Claude Code limit) and checked against the validation tables by a unit test.

### 6. One version, one bump command, drift caught by tests

The plugin manifests carry the package version. `scripts/set_version.py` sets it in `pyproject.toml`, `config.py`, the example config and the four manifests; `test_set_version.py` and `test_plugin_manifests.py` fail on any mismatch. The launcher is unpinned, so plugin and package versions are a bookkeeping convention, not a compatibility lock.

### 7. Validation in CI

The `plugin` job runs `claude plugin validate --strict ./plugins/pixelforge` and `claude plugin validate .` with the official CLI, validates the Cursor manifest and marketplace against Cursor's published JSON schemas (pinned commit), and builds an sdist to prove that `plugins/` and the marketplace files never reach PyPI. Codex has no validator; its manifest shape is covered by the unit tests. Pytest additionally checks frontmatter, line counts, reference links, model IDs and that no secret-looking strings appear in the skills.

## Consequences

- Install becomes one command on Claude Code and Codex and a repository URL on Cursor, with credentials entered through the client instead of a settings file.
- Existing Claude Code users who installed with `claude mcp add pixelforge` get a second server (38 tools) unless they remove the manual entry; the plugin README says so.
- Every model or pricing change already touches `validation.py`, `api_client.py`, the docstrings and the docs; the skills add one more place only for structural facts, which the tests cross-check against the validation tables.
- Official listings differ by vendor: Claude Code's self-hosted marketplace needs no submission; Codex's public directory is limited to remote servers or approved partners for local ones, so the Codex path is the Git marketplace; Cursor's official marketplace is blocked by the AGPL license.
- Publishing to the MCP Registry (`server.json`, README `mcp-name` marker) and adding MCP prompts were considered and deferred; both are independent of this layout.

## Alternatives considered

- **Server-side guidance only** (`instructions` plus MCP prompts, no plugin): reaches every client at the lowest cost, but gives no marketplace presence and no skills. Kept as part of this decision (item 5), not instead of it.
- **One Agent Plugins 1.0 manifest for all clients**: rejected; it disables the Codex overlay and cannot carry credentials for a stdio server.
- **Separate plugin repository**: would allow an MIT license for a Cursor listing, at the cost of skills drifting from the server and two repositories per release. Revisit if a Cursor listing becomes a goal.
- **Pinned `uvx pixelforge-mcp==X.Y.Z`**: reproducible, but turns every release into a plugin edit; rejected by the maintainers.
- **Bare `pixelforge-mcp`**: dedupes with existing manual installs, but requires a prior `pipx install`, which defeats the one-command install.

## References

- Claude Code plugin, marketplace and skills references (code.claude.com/docs/en/plugins, plugins-reference, plugin-marketplaces, skills)
- OpenAI Codex plugin documentation (developers.openai.com/plugins/build/plugins, codex/mcp) and the `openai/codex` sources for manifest discovery order, marketplace file order and the stdio environment allowlist
- Cursor plugin reference (cursor.com/docs/reference/plugins), the `cursor/plugins` JSON schemas and publisher terms
- Agent Skills specification (agentskills.io/specification)
- Google's Gemini image model documentation used by the skills (ai.google.dev/gemini-api/docs/image-generation, pricing, model cards)
