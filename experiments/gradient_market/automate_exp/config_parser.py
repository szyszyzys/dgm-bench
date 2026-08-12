import logging
from enum import Enum
from typing import Any

import yaml
from dacite import from_dict, Config as DaciteConfig, MissingValueError, WrongTypeError

from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig  # Assuming these are there

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _convert_lists_to_tuples_for_specific_fields(data: Any) -> Any:
    """
    Recursively converts lists to tuples for specific fields that are type-hinted as Tuple
    but often parsed as lists from YAML (e.g., trigger_shape).
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "trigger_shape" and isinstance(value, list):
                # Only convert if the field name is 'trigger_shape' and the value is a list
                data[key] = tuple(value)
            elif isinstance(value, (dict, list)):
                data[key] = _convert_lists_to_tuples_for_specific_fields(value)
    elif isinstance(data, list):
        # Recursively apply to elements in a list
        return [_convert_lists_to_tuples_for_specific_fields(item) for item in data]
    return data


def load_config(config_path: str) -> AppConfig:
    """
    Loads a YAML config file into a validated AppConfig dataclass.
    """
    logger.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)  # Use FullLoader
        processed_config_dict = _convert_lists_to_tuples_for_specific_fields(config_dict)

        # --- FIX: Added 'Enum' to the cast list ---
        # This tells dacite to automatically convert strings to Enum members
        type_config = DaciteConfig(cast=[Enum, bool, int, float, str])

        cfg = from_dict(
            data_class=AppConfig,
            data=processed_config_dict,
            config=type_config
        )

        logger.info("Successfully loaded and validated config.")
        return cfg

    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        raise
    except (MissingValueError, WrongTypeError) as e:
        logger.error(f"Error parsing config file '{config_path}': {e}")
        logger.error("Please check that your YAML file matches the structure of the AppConfig dataclasses.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the config: {e}")
        raise
