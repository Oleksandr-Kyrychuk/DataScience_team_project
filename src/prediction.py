import streamlit as st
import pandas as pd
import pickle
import logging
from inference import preprocess_input, predict_churn
from typing import Tuple, Union, Dict
from sklearn.preprocessing import StandardScaler
import numpy as np

logger = logging.getLogger(__name__)


@st.cache_resource
def load_model_and_scaler(config: Dict) -> Tuple[object, StandardScaler]:
    try:
        with open(config["paths"]["model"], "rb") as f:
            model = pickle.load(f)
        with open(config["paths"]["scaler"], "rb") as f:
            scaler = pickle.load(f)
        logger.info("Model and scaler loaded successfully.")
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        logger.error(f"File not found: {e}")
        st.stop()
    except pickle.UnpicklingError as e:
        st.error(f"Error deserializing file: {e}")
        logger.error(f"Error deserializing file: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unknown error loading model/scaler: {e}")
        logger.error(f"Unknown error loading model/scaler: {e}")
        st.stop()


def make_prediction(
    df: pd.DataFrame, config: Dict, data_source: str = "csv"
) -> np.ndarray:
    try:
        model, scaler = load_model_and_scaler(config)
        logger.info(f"DataFrame shape before preprocessing: {df.shape}")
        logger.info(f"Scaler type: {type(scaler)}")
        processed_data = preprocess_input(df, scaler, logger)
        predictions = predict_churn(model, processed_data, logger)
        predictions = predictions.tolist()  # Convert NumPy array to list
        logger.info(f"Prediction probabilities: {predictions}")
        logger.info(f"{data_source.capitalize()} predictions made successfully.")
        return predictions
    except Exception as e:
        st.error(
            f"{config['ui']['languages'][st.session_state.get('lang', 'en')]['predict_error']}: {e}"
        )
        logger.error(f"Error making {data_source} predictions: {e}")
        st.stop()
