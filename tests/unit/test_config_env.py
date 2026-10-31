"""Environment values that plugin hosts may leave empty or unexpanded."""

import pytest

from pixelforge_mcp.config import Config, env_value


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "${user_config.google_api_key}", "${GOOGLE_API_KEY}", "${}"],
)
def test_env_value_ignores_empty_and_placeholders(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("PIXELFORGE_TEST_VAR", raw)
    assert env_value("PIXELFORGE_TEST_VAR") is None


def test_env_value_returns_real_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIXELFORGE_TEST_VAR", "  abc  ")
    assert env_value("PIXELFORGE_TEST_VAR") == "abc"
    monkeypatch.delenv("PIXELFORGE_TEST_VAR")
    assert env_value("PIXELFORGE_TEST_VAR") is None


def test_placeholder_api_key_does_not_shadow_service_account(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    for name in ("GOOGLE_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(name, "${user_config.google_api_key}")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "example-project")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "${user_config.creds}")
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)

    config = Config.load(tmp_path / "missing.yaml")  # type: ignore[operator]

    assert config.imagen.api_key is None
    assert config.vertex.enabled is True
    assert config.backend == "vertex_ai"
    # The unexpanded credentials path was removed so google-auth falls back
    # to the other Application Default Credentials sources.
    import os

    assert "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ


def test_empty_project_keeps_gemini_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "real-key")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "")
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
    config = Config.load()
    assert config.vertex.enabled is False
    assert config.backend == "gemini_api"
