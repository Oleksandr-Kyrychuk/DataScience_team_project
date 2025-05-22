import streamlit as st
import yaml
import logging
from typing import Dict


def init_session_state() -> None:
    """Initialize Streamlit session state."""
    defaults = {
        "csv_data": None,
        "manual_data": None,
        "original_ids": None,
        "predictions": None,
        "lang": "en",  # Default language
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logging.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}")
        st.error(f"Configuration file not found: {e}")
        st.stop()
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config.yaml: {e}")
        st.error(f"Error parsing configuration file: {e}")
        st.stop()
