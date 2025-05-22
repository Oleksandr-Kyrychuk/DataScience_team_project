import pandas as pd
import numpy as np
import logging
from typing import Optional, Union
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import streamlit as st


@st.cache_data(hash_funcs={pd.DataFrame: lambda x: x.to_string()})
def preprocess_input(
    df: pd.DataFrame, _scaler: StandardScaler, _logger=None
) -> pd.DataFrame:
    if _logger is None:
        _logger = logging.getLogger(__name__)
    if df is None or df.empty:
        _logger.error("Input DataFrame cannot be empty or None.")
        raise ValueError("Input DataFrame cannot be empty or None.")

    def process_chunk(chunk):
        return chunk

    # Розбиття на чанки для великих даних
    if len(df) > 1000:
        n_chunks = (len(df) // 1000) + 1
        chunks = np.array_split(df, n_chunks)
        processed_chunks = Parallel(n_jobs=-1)(
            delayed(process_chunk)(chunk) for chunk in chunks
        )
        df = pd.concat(processed_chunks)
    else:
        df = process_chunk(df)

    required_cols = [
        "is_tv_subscriber",
        "is_movie_package_subscriber",
        "subscription_age",
        "remaining_contract",
        "service_failure_count",
        "download_avg",
        "upload_avg",
        "download_over_limit",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        _logger.warning(
            f"Missing columns: {missing_cols}. Filling with default values (0)."
        )
        for col in missing_cols:
            df[col] = 0

    if not pd.api.types.is_numeric_dtype(df["download_over_limit"]):
        _logger.error("Column 'download_over_limit' contains non-numeric values.")
        raise ValueError("Column 'download_over_limit' must contain numeric values.")

    df["download_over_limit"] = pd.to_numeric(
        df["download_over_limit"], errors="coerce"
    )
    df["download_over_limit"] = df["download_over_limit"].fillna(0).astype(int)
    df["download_over_limit"] = df["download_over_limit"].clip(0, 7)

    df["remaining_contract"] = df["remaining_contract"].fillna(0)
    df["download_avg"] = df["download_avg"].fillna(df["download_avg"].median())
    df["upload_avg"] = df["upload_avg"].fillna(df["upload_avg"].median())

    if (df["subscription_age"] < 0).any():
        _logger.warning(
            "Negative values found in 'subscription_age'. Replacing with median."
        )
        median_age = df.loc[df["subscription_age"] >= 0, "subscription_age"].median()
        df.loc[df["subscription_age"] < 0, "subscription_age"] = median_age

    Q1 = df["download_avg"].quantile(0.25)
    Q3 = df["download_avg"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df["download_avg"] = np.where(
        df["download_avg"] > upper,
        upper,
        np.where(df["download_avg"] < lower, lower, df["download_avg"]),
    )

    Q1 = df["upload_avg"].quantile(0.25)
    Q3 = df["upload_avg"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df["upload_avg"] = np.where(
        df["upload_avg"] > upper,
        upper,
        np.where(df["upload_avg"] < lower, lower, df["upload_avg"]),
    )

    for i in range(8):
        df[f"download_over_limit_{i}"] = (df["download_over_limit"] == i).astype(int)
    df.drop(columns=["download_over_limit"], errors="ignore", inplace=True)

    numeric_cols = [
        "subscription_age",
        "remaining_contract",
        "service_failure_count",
        "download_avg",
        "upload_avg",
    ]
    if _scaler is None:
        _logger.error("Scaler is required for preprocessing.")
        raise ValueError("Scaler is required for preprocessing.")
    df[numeric_cols] = _scaler.transform(df[numeric_cols])

    df.drop(columns=["id", "bill_avg"], errors="ignore", inplace=True)

    expected_columns = [
        "is_tv_subscriber",
        "is_movie_package_subscriber",
        "subscription_age",
        "remaining_contract",
        "service_failure_count",
        "download_avg",
        "upload_avg",
        "download_over_limit_0",
        "download_over_limit_1",
        "download_over_limit_2",
        "download_over_limit_3",
        "download_over_limit_4",
        "download_over_limit_5",
        "download_over_limit_6",
        "download_over_limit_7",
    ]
    df = df.reindex(columns=expected_columns, fill_value=0)
    _logger.info("Data successfully preprocessed for prediction.")
    return df


def predict_churn(
    model: object, data: pd.DataFrame, logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """Predict churn probabilities."""
    if logger is None:
        logger = logging.getLogger(__name__)

    if model is None:
        logger.error("Model cannot be None.")
        raise ValueError("Model cannot be None.")

    if data is None or data.empty:
        logger.error("Input data cannot be empty or None.")
        raise ValueError("Input data cannot be empty or None.")

    if set(data.columns) != set(model.feature_names_in_):
        logger.error("Input columns do not match model features.")
        raise ValueError("Input columns do not match model features.")

    try:
        predictions = model.predict_proba(data)[:, 1]
        logger.info("Predictions made successfully.")
        return predictions
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise ValueError(f"Error during prediction: {str(e)}")
