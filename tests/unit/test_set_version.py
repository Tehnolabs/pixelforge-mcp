"""scripts/set_version.py updates every recorded version together."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "set_version.py"


def load_module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("set_version", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repo_copy(tmp_path: Path) -> Path:
    module = load_module()
    for rel, _ in module.TEXT_LOCATIONS + module.JSON_LOCATIONS:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, target)
    return tmp_path


def test_sets_every_location(repo_copy: Path) -> None:
    module = load_module()
    changed = module.set_version(repo_copy, "9.8.7")
    assert all(changed.values()), changed
    assert 'version = "9.8.7"' in (repo_copy / "pyproject.toml").read_text()
    config_text = (repo_copy / "src/pixelforge_mcp/config.py").read_text()
    assert 'Field(default="9.8.7"' in config_text
    for rel, keys in module.JSON_LOCATIONS:
        node = json.loads((repo_copy / rel).read_text())
        for key in keys:
            node = node[key]
        assert node == "9.8.7", rel
    # Idempotent: a second run changes nothing.
    assert not any(module.set_version(repo_copy, "9.8.7").values())


def test_rejects_non_semver(repo_copy: Path) -> None:
    module = load_module()
    with pytest.raises(SystemExit):
        module.set_version(repo_copy, "latest")


def test_repository_versions_currently_agree() -> None:
    module = load_module()
    versions = set()
    for rel, pattern in module.TEXT_LOCATIONS:
        import re

        match = re.search(pattern, (ROOT / rel).read_text(), re.M)
        assert match, rel
        versions.add(match.group(2))
    for rel, keys in module.JSON_LOCATIONS:
        node = json.loads((ROOT / rel).read_text())
        for key in keys:
            node = node[key]
        versions.add(node)
    assert len(versions) == 1, versions
