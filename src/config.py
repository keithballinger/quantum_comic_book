"""
Configuration management for Quantum Comic Book Generator.

Handles environment variables and application settings.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Config:
    """Application configuration."""
    
    # API Keys
    ibm_api_key: str
    gemini_api_key: str
    
    # IBM Quantum settings
    ibm_backend: Optional[str] = None
    use_simulator: bool = False
    
    # Comic parameters
    panels: int = 6
    characters: int = 2
    
    # Random seed
    seed: Optional[int] = None
    
    # Output settings
    output_dir: Path = Path("output")
    
    # Runtime settings
    max_retries: int = 3
    timeout: int = 300  # seconds
    
    # Debug settings
    debug: bool = False
    dry_run: bool = False


class ConfigError(Exception):
    """Configuration error."""
    pass


def load_config(env_file: Optional[str] = None) -> Config:
    """
    Load configuration from environment variables.
    
    Args:
        env_file: Optional path to .env file
        
    Returns:
        Config object with validated settings
        
    Raises:
        ConfigError: If required environment variables are missing
    """
    # Load environment variables from .env file
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()
    
    # Required environment variables
    ibm_api_key = os.getenv("IBM_API_KEY")
    if not ibm_api_key:
        raise ConfigError(
            "IBM_API_KEY environment variable is required. "
            "Get your API key from: https://quantum.ibm.com/"
        )
    
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ConfigError(
            "GEMINI_API_KEY environment variable is required. "
            "Get your API key from: https://ai.google.dev/"
        )
    
    # Optional settings with defaults
    ibm_backend = os.getenv("IBM_BACKEND") or None
    use_simulator = os.getenv("USE_SIMULATOR", "false").lower() == "true"
    
    # Comic parameters
    panels = int(os.getenv("PANELS", "6"))
    if panels < 1 or panels > 12:
        raise ConfigError("PANELS must be between 1 and 12")
    
    characters = int(os.getenv("CHARACTERS", "2"))
    if characters < 1 or characters > 4:
        raise ConfigError("CHARACTERS must be between 1 and 4")
    
    # Random seed
    seed_str = os.getenv("SEED")
    seed = int(seed_str) if seed_str else None
    
    # Output directory
    output_dir = Path(os.getenv("OUTPUT_DIR", "output"))
    
    # Runtime settings
    max_retries = int(os.getenv("MAX_RETRIES", "3"))
    timeout = int(os.getenv("TIMEOUT", "300"))
    
    # Debug settings
    debug = os.getenv("DEBUG", "false").lower() == "true"
    dry_run = os.getenv("DRY_RUN", "false").lower() == "true"
    
    return Config(
        ibm_api_key=ibm_api_key,
        gemini_api_key=gemini_api_key,
        ibm_backend=ibm_backend,
        use_simulator=use_simulator,
        panels=panels,
        characters=characters,
        seed=seed,
        output_dir=output_dir,
        max_retries=max_retries,
        timeout=timeout,
        debug=debug,
        dry_run=dry_run,
    )


def validate_config(config: Config) -> None:
    """
    Validate configuration values.
    
    Args:
        config: Configuration object to validate
        
    Raises:
        ConfigError: If configuration is invalid
    """
    # Check qubit count doesn't exceed limits
    # Formula: T + 2*panels*3 + 4 (time + actions/emotions/camera + style)
    total_qubits = config.panels + (2 * config.panels * 3) + 4
    if total_qubits > 127:
        raise ConfigError(
            f"Configuration requires {total_qubits} qubits, "
            f"but maximum is 127. Reduce panels or characters."
        )
    
    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)


def get_circuit_parameters(config: Config) -> dict:
    """
    Get quantum circuit parameters from configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary with circuit parameters
    """
    return {
        "panels": config.panels,
        "characters": config.characters,
        "time_qubits": config.panels,
        "action_qubits": 2 * config.panels,
        "emotion_qubits": 2 * config.panels,
        "camera_qubits": 2 * config.panels,
        "style_qubits": 4,
        "total_qubits": config.panels + (2 * config.panels * 3) + 4,
        "seed": config.seed,
    }


def get_runtime_options(config: Config) -> dict:
    """
    Get IBM Runtime options from configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary with runtime options
    """
    return {
        "backend": config.ibm_backend,
        "use_simulator": config.use_simulator,
        "max_retries": config.max_retries,
        "timeout": config.timeout,
    }


# Default configuration instance
_default_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the default configuration instance.
    
    Returns:
        Default Config object
        
    Raises:
        ConfigError: If configuration hasn't been initialized
    """
    global _default_config
    if _default_config is None:
        _default_config = load_config()
        validate_config(_default_config)
    return _default_config


def set_config(config: Config) -> None:
    """
    Set the default configuration instance.
    
    Args:
        config: Configuration object to set as default
    """
    global _default_config
    validate_config(config)
    _default_config = config