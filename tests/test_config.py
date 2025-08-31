"""
Tests for configuration management.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import (
    Config,
    ConfigError,
    get_circuit_parameters,
    get_runtime_options,
    load_config,
    validate_config,
)


class TestConfig:
    """Test configuration loading and validation."""

    def test_load_config_missing_ibm_key(self):
        """Test that missing IBM_API_KEY raises ConfigError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigError, match="IBM_API_KEY"):
                load_config()

    def test_load_config_missing_gemini_key(self):
        """Test that missing GEMINI_API_KEY raises ConfigError."""
        with patch.dict(os.environ, {"IBM_API_KEY": "test_key"}, clear=True):
            with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
                load_config()

    def test_load_config_success(self):
        """Test successful configuration loading with required keys."""
        env_vars = {
            "IBM_API_KEY": "test_ibm_key",
            "GEMINI_API_KEY": "test_gemini_key",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = load_config()
            assert config.ibm_api_key == "test_ibm_key"
            assert config.gemini_api_key == "test_gemini_key"
            assert config.panels == 6  # default
            assert config.characters == 2  # default

    def test_load_config_with_optional_settings(self):
        """Test configuration loading with optional settings."""
        env_vars = {
            "IBM_API_KEY": "test_ibm_key",
            "GEMINI_API_KEY": "test_gemini_key",
            "IBM_BACKEND": "ibm_brisbane",
            "USE_SIMULATOR": "true",
            "PANELS": "8",
            "CHARACTERS": "3",
            "SEED": "42",
            "DEBUG": "true",
            "DRY_RUN": "true",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = load_config()
            assert config.ibm_backend == "ibm_brisbane"
            assert config.use_simulator is True
            assert config.panels == 8
            assert config.characters == 3
            assert config.seed == 42
            assert config.debug is True
            assert config.dry_run is True

    def test_load_config_from_env_file(self):
        """Test loading configuration from .env file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("IBM_API_KEY=file_ibm_key\n")
            f.write("GEMINI_API_KEY=file_gemini_key\n")
            f.write("PANELS=4\n")
            env_file = f.name

        try:
            with patch.dict(os.environ, {}, clear=True):
                config = load_config(env_file)
                assert config.ibm_api_key == "file_ibm_key"
                assert config.gemini_api_key == "file_gemini_key"
                assert config.panels == 4
        finally:
            os.unlink(env_file)

    def test_panels_validation(self):
        """Test panels validation."""
        env_vars = {
            "IBM_API_KEY": "test_key",
            "GEMINI_API_KEY": "test_key",
        }

        # Test invalid panels (too few)
        with patch.dict(os.environ, {**env_vars, "PANELS": "0"}, clear=True):
            with pytest.raises(ConfigError, match="PANELS must be between"):
                load_config()

        # Test invalid panels (too many)
        with patch.dict(os.environ, {**env_vars, "PANELS": "13"}, clear=True):
            with pytest.raises(ConfigError, match="PANELS must be between"):
                load_config()

    def test_characters_validation(self):
        """Test characters validation."""
        env_vars = {
            "IBM_API_KEY": "test_key",
            "GEMINI_API_KEY": "test_key",
        }

        # Test invalid characters (too few)
        with patch.dict(os.environ, {**env_vars, "CHARACTERS": "0"}, clear=True):
            with pytest.raises(ConfigError, match="CHARACTERS must be between"):
                load_config()

        # Test invalid characters (too many)
        with patch.dict(os.environ, {**env_vars, "CHARACTERS": "5"}, clear=True):
            with pytest.raises(ConfigError, match="CHARACTERS must be between"):
                load_config()


class TestValidateConfig:
    """Test configuration validation."""

    def test_validate_config_qubit_limit(self):
        """Test that configuration with too many qubits raises error."""
        config = Config(
            ibm_api_key="test",
            gemini_api_key="test",
            panels=20,  # This will exceed 127 qubits
            characters=2,
        )
        with pytest.raises(ConfigError, match="requires .* qubits"):
            validate_config(config)

    def test_validate_config_creates_output_dir(self, tmp_path):
        """Test that validation creates output directory."""
        output_dir = tmp_path / "test_output"
        config = Config(
            ibm_api_key="test",
            gemini_api_key="test",
            output_dir=output_dir,
        )

        assert not output_dir.exists()
        validate_config(config)
        assert output_dir.exists()


class TestGetCircuitParameters:
    """Test circuit parameter extraction."""

    def test_get_circuit_parameters(self):
        """Test getting circuit parameters from config."""
        config = Config(
            ibm_api_key="test",
            gemini_api_key="test",
            panels=6,
            characters=2,
            seed=42,
        )

        params = get_circuit_parameters(config)

        assert params["panels"] == 6
        assert params["characters"] == 2
        assert params["time_qubits"] == 6
        assert params["action_qubits"] == 12
        assert params["emotion_qubits"] == 12
        assert params["camera_qubits"] == 12
        assert params["style_qubits"] == 4
        assert params["total_qubits"] == 46
        assert params["seed"] == 42


class TestGetRuntimeOptions:
    """Test runtime options extraction."""

    def test_get_runtime_options(self):
        """Test getting runtime options from config."""
        config = Config(
            ibm_api_key="test",
            gemini_api_key="test",
            ibm_backend="ibm_brisbane",
            use_simulator=True,
            max_retries=5,
            timeout=600,
        )

        options = get_runtime_options(config)

        assert options["backend"] == "ibm_brisbane"
        assert options["use_simulator"] is True
        assert options["max_retries"] == 5
        assert options["timeout"] == 600


# Parametrized tests for different configurations
@pytest.mark.parametrize(
    "panels,characters,expected_qubits",
    [
        (1, 1, 11),  # Minimum configuration
        (6, 2, 46),  # Default configuration
        (12, 4, 88),  # Maximum typical configuration
    ],
)
def test_qubit_calculation(panels, characters, expected_qubits):
    """Test qubit count calculation for various configurations."""
    config = Config(
        ibm_api_key="test",
        gemini_api_key="test",
        panels=panels,
        characters=characters,
    )

    params = get_circuit_parameters(config)
    assert params["total_qubits"] == expected_qubits
