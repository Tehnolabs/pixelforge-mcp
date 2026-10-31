#!/usr/bin/env python3
"""Set the project version everywhere it is recorded.

Usage: python scripts/set_version.py X.Y.Z

The version lives in the package metadata, the server config default, the
example config, and the plugin manifests that ship with the repository.
The unit tests fail when any of them disagree, so always bump with this
script rather than by hand.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.\-]+)?$")

# (relative path, regex with one group around the version)
TEXT_LOCATIONS: List[Tuple[str, str]] = [
    ("pyproject.toml", r'^(version = ")([^"]+)(")'),
    ("src/pixelforge_mcp/config.py", r'(version: str = Field\(default=")([^"]+)(")'),
    ("config/config.example.yaml", r'^(  version: ")([^"]+)(")'),
]

# (relative path, key path inside the JSON document)
JSON_LOCATIONS: List[Tuple[str, Tuple[str, ...]]] = [
    ("plugins/pixelforge/.claude-plugin/plugin.json", ("version",)),
    ("plugins/pixelforge/.codex-plugin/plugin.json", ("version",)),
    ("plugins/pixelforge/.cursor-plugin/plugin.json", ("version",)),
    (".claude-plugin/marketplace.json", ("metadata", "version")),
]


def set_text_version(path: Path, pattern: str, version: str) -> bool:
    text = path.read_text(encoding="utf-8")
    new_text, count = re.subn(
        pattern, lambda m: f"{m.group(1)}{version}{m.group(3)}", text, flags=re.M
    )
    if count != 1:
        raise SystemExit(f"{path}: expected one version match, found {count}")
    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
        return True
    return False


def set_json_version(path: Path, keys: Tuple[str, ...], version: str) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    node = data
    for key in keys[:-1]:
        node = node[key]
    if node.get(keys[-1]) == version:
        return False
    node[keys[-1]] = version
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return True


def set_version(root: Path, version: str) -> Dict[str, bool]:
    if not SEMVER.match(version):
        raise SystemExit(f"not a semantic version: {version}")
    changed: Dict[str, bool] = {}
    for rel, pattern in TEXT_LOCATIONS:
        changed[rel] = set_text_version(root / rel, pattern, version)
    for rel, keys in JSON_LOCATIONS:
        changed[rel] = set_json_version(root / rel, keys, version)
    return changed


def main(argv: List[str], report: Callable[[str], None] = print) -> int:
    if len(argv) != 2:
        report(__doc__ or "")
        return 2
    for rel, changed in set_version(ROOT, argv[1]).items():
        report(f"{'updated' if changed else 'already'}  {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
