# src/utils/config_parser.py

import yaml
from typing import Dict, Any

def load_config(path: str) -> Dict[str, Any]:
    """Load and parse a YAML configuration file.

    Args:
        path: The path to the YAML configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration parameters.
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config