"""
Configuration loading and management.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict[str, Any], path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge two configuration dictionaries.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.

    Returns:
        Default configuration dictionary
    """
    return {
        'general': {
            'seed': 42,
            'device': 'auto',
        },
        'topology': {
            'watts_strogatz': {
                'n_nodes': 100,
                'k_neighbors': 4,
                'beta': 0.3,
            },
        },
        'training': {
            'batch_size': 64,
            'learning_rate': 0.001,
            'epochs': 100,
        },
    }
