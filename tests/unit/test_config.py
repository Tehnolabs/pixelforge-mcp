"""Unit tests for configuration management."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from pixelforge_mcp.config import (
    Config,
    ImagenConfig,
    ServerConfig,
    StorageConfig,
    get_config,
    reload_config,
)


class TestImagenConfig:
    """Tests for ImagenConfig model."""

    def test_default_values(self):
        """Test ImagenConfig uses correct defaults."""
        config = ImagenConfig()
        assert config.default_model == "gemini-2.5-flash-image"
        assert config.default_aspect_ratio == "1:1"
        assert config.default_temperature == 0.7
        assert config.safety_setting == "preset:strict"
        assert config.api_key is None

    def test_temperature_validation_valid(self):
        """Test temperature validation accepts valid values."""
        config = ImagenConfig(default_temperature=0.0)
        assert config.default_temperature == 0.0

        config = ImagenConfig(default_temperature=1.0)
        assert config.default_temperature == 1.0

        config = ImagenConfig(default_temperature=0.5)
        assert config.default_temperature == 0.5

    def test_temperature_validation_invalid(self):
        """Test temperature validation rejects invalid values."""
        with pytest.raises(ValueError, match="Temperature must be between 0 and 1"):
            ImagenConfig(default_temperature=-0.1)

        with pytest.raises(ValueError, match="Temperature must be between 0 and 1"):
            ImagenConfig(default_temperature=1.1)


class TestStorageConfig:
    """Tests for StorageConfig model."""

    def test_default_values(self):
        """Test StorageConfig uses correct defaults."""
        config = StorageConfig()
        assert config.output_dir == Path("./generated_images")
        assert config.use_s3 is False
        assert config.s3_bucket is None
        assert config.s3_prefix == ""

    def test_output_dir_creation(self, tmp_path):
        """Test output directory is created if it doesn't exist."""
        output_dir = tmp_path / "test_images"
        assert not output_dir.exists()

        config = StorageConfig(output_dir=output_dir)
        assert output_dir.exists()
        assert output_dir.is_dir()


class TestServerConfig:
    """Tests for ServerConfig model."""

    def test_default_values(self):
        """Test ServerConfig uses correct defaults."""
        config = ServerConfig()
        assert config.name == "gemini-imagen-mcp"
        assert config.version == "0.1.0"
        assert config.log_level == "INFO"


class TestConfig:
    """Tests for main Config model."""

    def test_default_values(self):
        """Test Config uses correct defaults."""
        config = Config()
        assert isinstance(config.imagen, ImagenConfig)
        assert isinstance(config.storage, StorageConfig)
        assert isinstance(config.server, ServerConfig)

    def test_load_from_empty_file(self, tmp_path):
        """Test loading config from empty YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        config = Config.load(config_file)
        assert isinstance(config, Config)
        assert config.server.name == "gemini-imagen-mcp"

    def test_load_from_yaml_file(self, tmp_path):
        """Test loading config from YAML file."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "imagen": {
                "api_key": "test-key",
                "default_model": "custom-model",
                "default_temperature": 0.9,
            },
            "storage": {
                "output_dir": str(tmp_path / "images"),
                "use_s3": True,
                "s3_bucket": "test-bucket",
            },
            "server": {
                "log_level": "DEBUG",
            },
        }
        config_file.write_text(yaml.dump(config_data))

        config = Config.load(config_file)
        assert config.imagen.api_key == "test-key"
        assert config.imagen.default_model == "custom-model"
        assert config.imagen.default_temperature == 0.9
        assert config.storage.use_s3 is True
        assert config.storage.s3_bucket == "test-bucket"
        assert config.server.log_level == "DEBUG"

    def test_load_with_env_override(self, tmp_path, monkeypatch):
        """Test environment variables override config file."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "imagen": {
                "api_key": "file-key",
            }
        }
        config_file.write_text(yaml.dump(config_data))

        # Set environment variable
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")

        config = Config.load(config_file)
        assert config.imagen.api_key == "env-key"

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading from non-existent file returns defaults."""
        config_file = tmp_path / "nonexistent.yaml"
        config = Config.load(config_file)
        assert isinstance(config, Config)
        assert config.server.name == "gemini-imagen-mcp"

    def test_save_config(self, tmp_path):
        """Test saving config to file."""
        config_file = tmp_path / "config.yaml"

        config = Config(
            imagen=ImagenConfig(
                api_key="test-key",
                default_temperature=0.8,
            ),
            server=ServerConfig(log_level="DEBUG"),
        )

        config.save(config_file)

        # Verify file was created
        assert config_file.exists()

        # Load and verify contents
        loaded_config = Config.load(config_file)
        assert loaded_config.imagen.api_key == "test-key"
        assert loaded_config.imagen.default_temperature == 0.8
        assert loaded_config.server.log_level == "DEBUG"


class TestConfigSingleton:
    """Tests for global config instance."""

    def test_get_config_returns_instance(self):
        """Test get_config returns a Config instance."""
        config = get_config()
        assert isinstance(config, Config)

    def test_get_config_returns_same_instance(self):
        """Test get_config returns the same instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_reload_config(self, tmp_path):
        """Test reload_config reloads from file."""
        # Import the module to reset the global
        import pixelforge_mcp.config as config_module

        # Reset the global config
        config_module._config = None

        # First get config
        config1 = get_config()

        # Reset again and create a different config
        config_module._config = None

        # Call reload_config
        reload_config()
        config2 = get_config()

        # Should be different instance
        assert config2 is not config1
        # But same type
        assert isinstance(config2, Config)
