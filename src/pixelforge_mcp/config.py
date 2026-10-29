"""Configuration management for the PixelForge MCP server."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator


class MaskedStr(SecretStr):
    """A string that masks its value in repr/str.

    Supports equality comparison with plain strings.
    """

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SecretStr):
            return self.get_secret_value() == other.get_secret_value()
        if isinstance(other, str):
            return self.get_secret_value() == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.get_secret_value())


class ImagenConfig(BaseModel):
    """Image generation defaults."""

    api_key: Optional[MaskedStr] = Field(default=None, description="Google API key")
    default_model: str = Field(
        default="gemini-nano-banana-2.1",
        description="Default image generation model",
    )
    default_aspect_ratio: str = Field(default="1:1", description="Default aspect ratio")
    default_temperature: float = Field(default=0.7, description="Default temperature")
    safety_setting: str = Field(
        default="preset:strict", description="Safety filter preset"
    )

    @field_validator("default_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Ensure temperature is between 0 and 2."""
        if not 0 <= v <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v


class StorageConfig(BaseModel):
    """Storage configuration."""

    output_dir: Path = Field(
        default=Path("./generated_images"),
        description="Directory for generated images",
    )

    @field_validator("output_dir")
    @classmethod
    def create_output_dir(cls, v: Path) -> Path:
        """Create output directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v


class ServerConfig(BaseModel):
    """MCP server configuration."""

    name: str = Field(default="pixelforge-mcp", description="Server name")
    version: str = Field(default="0.6.0", description="Server version")
    log_level: str = Field(default="INFO", description="Logging level")


BACKEND_GEMINI_API = "gemini_api"
BACKEND_VERTEX_AI = "vertex_ai"

_TRUTHY = {"1", "true", "yes", "on"}

# Switches the google-genai SDK itself reads to route requests to Vertex AI.
# GOOGLE_GENAI_USE_ENTERPRISE is the newer name; both are accepted.
VERTEX_SWITCH_ENV_VARS = ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_ENTERPRISE")


def env_flag(name: str) -> bool:
    """True when an environment variable holds 1, true, yes, or on."""
    return os.getenv(name, "").strip().lower() in _TRUTHY


def prefer_vertex_from_env() -> bool:
    """True when either SDK switch asks for Vertex AI."""
    return any(env_flag(name) for name in VERTEX_SWITCH_ENV_VARS)


class VertexConfig(BaseModel):
    """Optional Vertex AI backend configuration.

    When a Google Cloud project is configured and Application Default
    Credentials are available (for example a service account key in
    ``GOOGLE_APPLICATION_CREDENTIALS``), every tool can run through
    Vertex AI instead of the Gemini API key.
    """

    project_id: Optional[str] = Field(
        default=None, description="Google Cloud project ID"
    )
    location: str = Field(
        default="global",
        description="Vertex AI location. The current Gemini image and text "
        "models are served from the global endpoint.",
    )
    enabled: bool = Field(
        default=False, description="Whether Vertex AI credentials are configured"
    )
    prefer: bool = Field(
        default=False,
        description="Use Vertex AI even when an API key is also configured "
        "(GOOGLE_GENAI_USE_VERTEXAI=true or GOOGLE_GENAI_USE_ENTERPRISE=true)",
    )


class Config(BaseModel):
    """Main configuration."""

    imagen: ImagenConfig = Field(default_factory=lambda: ImagenConfig())
    storage: StorageConfig = Field(default_factory=lambda: StorageConfig())
    server: ServerConfig = Field(default_factory=lambda: ServerConfig())
    vertex: VertexConfig = Field(default_factory=lambda: VertexConfig())

    @property
    def backend(self) -> str:
        """Which Google backend serves requests.

        An API key selects the Gemini API. Vertex AI is used when it is
        explicitly preferred, or when a project is configured and no API
        key is set.
        """
        if self.vertex.prefer:
            return BACKEND_VERTEX_AI
        if self.vertex.enabled and not self.imagen.api_key:
            return BACKEND_VERTEX_AI
        return BACKEND_GEMINI_API

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Config":
        """Load configuration from file and environment variables."""
        # Default config path
        if config_path is None:
            config_path = Path("config/config.yaml")

        # Load from file if exists
        config_data: Dict[str, Any] = {}
        if config_path.exists():
            with open(config_path) as f:
                config_data = yaml.safe_load(f) or {}

        # Override with environment variables
        # Try multiple common env var names
        api_key = (
            os.getenv("GOOGLE_API_KEY")
            or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
        if api_key:
            config_data.setdefault("imagen", {})["api_key"] = api_key

        # Auto-detect Vertex AI. These are the same variables the
        # google-genai SDK and google-github-actions/auth use.
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if project_id:
            config_data.setdefault("vertex", {})["project_id"] = project_id
            config_data["vertex"]["enabled"] = True
        location = os.getenv("GOOGLE_CLOUD_LOCATION")
        if location:
            config_data.setdefault("vertex", {})["location"] = location
        if prefer_vertex_from_env():
            config_data.setdefault("vertex", {})["prefer"] = True

        return cls(**config_data)

    def save(self, config_path: Optional[Path] = None) -> None:
        """Save configuration to file."""
        if config_path is None:
            config_path = Path("config/config.yaml")

        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict with proper serialization
        data = self.model_dump()
        # Never persist API keys to disk — they should come from env vars
        if "imagen" in data:
            data["imagen"].pop("api_key", None)
        # Convert Path objects to strings
        if "storage" in data and "output_dir" in data["storage"]:
            data["storage"]["output_dir"] = str(data["storage"]["output_dir"])

        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def reload_config() -> None:
    """Reload configuration from file."""
    global _config
    _config = Config.load()
