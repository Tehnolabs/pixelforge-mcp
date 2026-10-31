"""The plugin bundle under plugins/pixelforge stays consistent and portable.

Three clients read the same bundle: Claude Code (.claude-plugin), Codex
(.codex-plugin) and Cursor (.cursor-plugin). Each has its own MCP file
because they expand environment placeholders differently. These checks
catch drift between the manifests, the marketplace files at the repo root,
the package version and the shared skills.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from pixelforge_mcp.server import mcp
from pixelforge_mcp.utils.validation import ANALYSIS_MODELS, IMAGE_MODELS

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "plugins" / "pixelforge"
SKILLS = PLUGIN / "skills"
LAUNCHER = {"command": "uvx", "args": ["pixelforge-mcp@latest"]}
CREDENTIAL_VARS = {
    "GOOGLE_API_KEY",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_APPLICATION_CREDENTIALS",
}
SPEC_FRONTMATTER = {"name", "description", "license", "compatibility", "metadata"}
ALLOWED_FRONTMATTER = SPEC_FRONTMATTER | {"allowed-tools"}
CLAUDE_TOOL_PREFIX = "mcp__plugin_pixelforge_pixelforge__"
SECRET_PATTERN = re.compile(r"\b(AIza[0-9A-Za-z_\-]{30,}|sk-[0-9A-Za-z]{20,})\b")
EXPECTED_SKILLS = {
    "pixelforge-setup",
    "pixelforge-generate",
    "pixelforge-edit",
    "pixelforge-analyze",
    "pixelforge-web-assets",
    "pixelforge-diagrams",
    "pixelforge-characters",
}


def load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    assert isinstance(data, dict)
    return data


def package_version() -> str:
    text = (ROOT / "pyproject.toml").read_text()
    match = re.search(r'^version = "([^"]+)"', text, re.MULTILINE)
    assert match
    return match.group(1)


@pytest.fixture(scope="module")
def manifests() -> Dict[str, Dict[str, Any]]:
    return {
        "claude": load_json(PLUGIN / ".claude-plugin" / "plugin.json"),
        "codex": load_json(PLUGIN / ".codex-plugin" / "plugin.json"),
        "cursor": load_json(PLUGIN / ".cursor-plugin" / "plugin.json"),
    }


@pytest.fixture(scope="module")
def mcp_files() -> Dict[str, Dict[str, Any]]:
    return {
        name: load_json(PLUGIN / "mcp" / f"{name}.json")
        for name in ("claude", "codex", "cursor")
    }


def frontmatter(skill_md: Path) -> Dict[str, Any]:
    text = skill_md.read_text()
    assert text.startswith("---\n"), f"{skill_md}: frontmatter must start on line 1"
    end = text.index("\n---", 4)
    data = yaml.safe_load(text[4:end])
    assert isinstance(data, dict), f"{skill_md}: frontmatter is not a mapping"
    return data


def skill_dirs() -> List[Path]:
    return sorted(p for p in SKILLS.iterdir() if p.is_dir())


# --- manifests ----------------------------------------------------------


def test_plugin_names_match(manifests: Dict[str, Dict[str, Any]]) -> None:
    assert {m["name"] for m in manifests.values()} == {"pixelforge"}


def test_plugin_versions_track_the_package(
    manifests: Dict[str, Dict[str, Any]],
) -> None:
    version = package_version()
    for client, manifest in manifests.items():
        assert manifest["version"] == version, f"{client} manifest version drifted"
    claude_market = load_json(ROOT / ".claude-plugin" / "marketplace.json")
    assert claude_market["metadata"]["version"] == version


def test_each_manifest_points_at_its_own_mcp_file(
    manifests: Dict[str, Dict[str, Any]],
) -> None:
    for client, manifest in manifests.items():
        assert manifest["mcpServers"] == f"./mcp/{client}.json"
        assert manifest["skills"] == "./skills/"
    assert not (PLUGIN / ".mcp.json").exists(), "a shared .mcp.json is read twice"
    assert not (PLUGIN / "mcp.json").exists(), "a root mcp.json is read by Cursor"
    assert not (PLUGIN / "plugin.json").exists(), "a root plugin.json blinds Codex"
    assert not (PLUGIN / "CLAUDE.md").exists()


def test_launcher_is_identical_and_unpinned(
    mcp_files: Dict[str, Dict[str, Any]],
) -> None:
    for client, data in mcp_files.items():
        assert list(data["mcpServers"]) == ["pixelforge"], client
        server = data["mcpServers"]["pixelforge"]
        assert server["command"] == LAUNCHER["command"], client
        assert server["args"] == LAUNCHER["args"], client


def test_claude_env_placeholders_are_declared(
    manifests: Dict[str, Dict[str, Any]], mcp_files: Dict[str, Dict[str, Any]]
) -> None:
    env = mcp_files["claude"]["mcpServers"]["pixelforge"]["env"]
    user_config = manifests["claude"]["userConfig"]
    assert set(env) == CREDENTIAL_VARS
    for name, value in env.items():
        # Shell export first, then the plugin setting, else empty. The
        # config-first nesting makes Claude Code drop the server (probed
        # on 2026-10-31), so the order here is load-bearing.
        match = re.fullmatch(r"\$\{([A-Z_]+):-\$\{user_config\.([a-z_]+)\}\}", value)
        assert match, f"Claude env must be ${{VAR:-${{user_config.key}}}}: {value}"
        assert match.group(1) == name
        assert match.group(2) in user_config
    key = user_config["google_api_key"]
    assert key["sensitive"] is True and key["required"] is False
    assert user_config["google_application_credentials"]["type"] == "file"


def test_codex_forwards_credentials_through_env_vars(
    mcp_files: Dict[str, Dict[str, Any]],
) -> None:
    server = mcp_files["codex"]["mcpServers"]["pixelforge"]
    assert "env" not in server, "a static env block would hold committed secrets"
    assert CREDENTIAL_VARS <= set(server["env_vars"])
    assert server["startup_timeout_sec"] >= 60, "a cold uvx resolve needs headroom"


def test_cursor_env_placeholders_are_declared_variables(
    manifests: Dict[str, Dict[str, Any]], mcp_files: Dict[str, Dict[str, Any]]
) -> None:
    server = mcp_files["cursor"]["mcpServers"]["pixelforge"]
    assert server["type"] == "stdio"
    variables = manifests["cursor"]["variables"]
    assert variables["type"] == "object"
    assert set(server["env"]) == CREDENTIAL_VARS
    for name, value in server["env"].items():
        assert value == f"${{{name}}}", f"not a plugin variable placeholder: {value}"
        assert name in variables["properties"]
    assert "required" not in variables, "credentials are optional (key or account)"
    assert "$schema" not in manifests["cursor"], "Cursor rejects unknown top-level keys"


def test_codex_manifest_shape(manifests: Dict[str, Dict[str, Any]]) -> None:
    codex = manifests["codex"]
    assert "$schema" not in codex
    onboarding = codex["extensions"]["com.openai"]["onboardingSkill"]
    assert onboarding == "./skills/pixelforge-setup/SKILL.md"
    interface = codex["interface"]
    assert len(interface["displayName"]) <= 30
    assert len(interface["shortDescription"]) <= 30
    assert 1 <= len(interface["defaultPrompt"]) <= 3
    assert all(len(p) <= 128 for p in interface["defaultPrompt"])
    for key in ("composerIcon", "logo"):
        assert (PLUGIN / interface[key]).is_file(), key


def test_manifest_paths_exist(manifests: Dict[str, Dict[str, Any]]) -> None:
    for client, manifest in manifests.items():
        for key in ("skills", "mcpServers"):
            assert (PLUGIN / manifest[key]).exists(), f"{client}: {key} missing"
    assert (PLUGIN / manifests["cursor"]["logo"]).is_file()


def test_descriptions_are_within_limits(manifests: Dict[str, Dict[str, Any]]) -> None:
    for client, manifest in manifests.items():
        assert 40 <= len(manifest["description"]) <= 1024, client


def test_plugin_readme_is_substantial() -> None:
    words = (PLUGIN / "README.md").read_text().split()
    assert len(words) >= 40


# --- marketplaces -------------------------------------------------------


def test_marketplaces_point_at_the_plugin_directory() -> None:
    claude = load_json(ROOT / ".claude-plugin" / "marketplace.json")
    assert claude["name"] == "pixelforge"
    assert claude["plugins"][0]["name"] == "pixelforge"
    assert claude["plugins"][0]["source"] == "./plugins/pixelforge"

    codex = load_json(ROOT / ".agents" / "plugins" / "marketplace.json")
    assert codex["name"] == "pixelforge"
    entry = codex["plugins"][0]
    assert entry["name"] == "pixelforge"
    assert entry["source"] == {"source": "local", "path": "./plugins/pixelforge"}
    assert entry["policy"]["installation"] in {"AVAILABLE", "INSTALLED_BY_DEFAULT"}
    assert entry["policy"]["authentication"] in {"ON_INSTALL", "ON_USE"}

    cursor = load_json(ROOT / ".cursor-plugin" / "marketplace.json")
    assert cursor["name"] == "pixelforge"
    assert cursor["plugins"][0]["name"] == "pixelforge"
    assert cursor["plugins"][0]["source"] == "./plugins/pixelforge"
    assert "owner" in cursor


# --- skills -------------------------------------------------------------


def test_expected_skills_are_present() -> None:
    assert {p.name for p in skill_dirs()} == EXPECTED_SKILLS


@pytest.mark.parametrize("skill", sorted(EXPECTED_SKILLS))
def test_skill_frontmatter_follows_the_open_standard(skill: str) -> None:
    skill_md = SKILLS / skill / "SKILL.md"
    assert skill_md.is_file()
    data = frontmatter(skill_md)
    unknown = set(data) - ALLOWED_FRONTMATTER
    assert not unknown, f"{skill}: unknown frontmatter fields {unknown}"
    assert data["name"] == skill, "name must equal the directory name"
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", data["name"])
    description = data["description"]
    assert isinstance(description, str) and 1 <= len(description) <= 1024
    assert "<" not in description, "no XML tags in descriptions"
    if "allowed-tools" in data:
        tools = {tool.name for tool in asyncio.run(mcp.list_tools())}
        for entry in str(data["allowed-tools"]).split():
            assert entry.startswith(CLAUDE_TOOL_PREFIX), entry
            assert entry[len(CLAUDE_TOOL_PREFIX) :] in tools, entry


@pytest.mark.parametrize("skill", sorted(EXPECTED_SKILLS))
def test_skill_body_is_short_and_self_contained(skill: str) -> None:
    skill_md = SKILLS / skill / "SKILL.md"
    lines = skill_md.read_text().splitlines()
    assert len(lines) <= 200, f"{skill}: SKILL.md has {len(lines)} lines"
    text = "\n".join(lines)
    body = text.split("\n---", 2)[-1]
    assert "mcp__" not in body, f"{skill}: no client prefixes in the body"
    assert not SECRET_PATTERN.search(text)
    refs_dir = SKILLS / skill / "references"
    mentioned = set(re.findall(r"references/([A-Za-z0-9_\-]+\.md)", body))
    existing = {p.name for p in refs_dir.glob("*.md")} if refs_dir.is_dir() else set()
    assert mentioned == existing, f"{skill}: references mismatch {mentioned ^ existing}"
    for ref in existing:
        ref_lines = (refs_dir / ref).read_text().splitlines()
        assert len(ref_lines) <= 400, f"{skill}/references/{ref} is too long"
    for path in (SKILLS / skill).rglob("*"):
        depth = len(path.relative_to(SKILLS / skill).parts)
        assert depth <= 2, f"{path}: references must stay one level deep"


@pytest.mark.parametrize("skill", sorted(EXPECTED_SKILLS))
def test_skill_mentions_only_real_models(skill: str) -> None:
    known = set(IMAGE_MODELS) | set(ANALYSIS_MODELS)
    for path in (SKILLS / skill).rglob("*.md"):
        # Model IDs start with a digit or "nano"; "gemini-api" in URLs is not one.
        found = re.findall(r"gemini-(?:\d|nano)[a-z0-9.\-]*", path.read_text())
        mentioned = {m.rstrip(".,;:)") for m in found}
        unknown = mentioned - known
        assert not unknown, f"{path}: unknown model ids {unknown}"
