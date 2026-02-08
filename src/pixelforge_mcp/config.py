"""Configuration management for Gemini Imagen MCP Server."""

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class ImagenConfig(BaseModel):
    """Gemini Imagen CLI configuration."""

    api_key: Optional[str] = Field(None, description="Google API key")
    default_model: str = Field(
        "gemini-2.5-flash-image",
        description="Default image generation model"
    )
    default_aspect_ratio: str = Field("1:1", description="Default aspect ratio")
    default_temperature: float = Field(0.7, description="Default temperature")
    safety_setting: str = Field("preset:strict", description="Safety filter preset")

    @field_validator("default_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Ensure temperature is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        return v


class StorageConfig(BaseModel):
    """Storage configuration."""

    output_dir: Path = Field(
        Path("./generated_images"),
        description="Directory for generated images"
    )
    use_s3: bool = Field(False, description="Enable S3 storage")
    s3_bucket: Optional[str] = Field(None, description="S3 bucket name")
    s3_prefix: str = Field("", description="S3 key prefix")

    @field_validator("output_dir")
    @classmethod
    def create_output_dir(cls, v: Path) -> Path:
        """Create output directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v


class ServerConfig(BaseModel):
    """MCP server configuration."""

    name: str = Field("gemini-imagen-mcp", description="Server name")
    version: str = Field("0.1.4", description="Server version")
    log_level: str = Field("INFO", description="Logging level")


class Config(BaseModel):
    """Main configuration."""

    imagen: ImagenConfig = Field(default_factory=ImagenConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Config":
        """Load configuration from file and environment variables."""
        # Default config path
        if config_path is None:
            config_path = Path("config/config.yaml")

        # Load from file if exists
        config_data = {}
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

        return cls(**config_data)

    def save(self, config_path: Optional[Path] = None) -> None:
        """Save configuration to file."""
        if config_path is None:
            config_path = Path("config/config.yaml")

        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict with proper serialization
        data = self.model_dump()
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
