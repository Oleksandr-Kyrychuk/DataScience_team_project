import pytest
import pandas as pd
from src.preprocessing import preprocess_data


def test_preprocess_data_missing_columns():
    """Test preprocessing with missing columns."""
    df = pd.DataFrame({"id": [1], "subscription_age": [2.5]})
    processed_df, scaler = preprocess_data(df=df, return_scaler=True)
    expected_cols = [
        "subscription_age",
        "reamining_contract",
        "service_failure_count",
        "download_avg",
        "upload_avg",
        "is_tv_subscriber",
        "is_movie_package_subscriber",
    ]
    assert all(col in processed_df.columns for col in expected_cols)
    assert processed_df.shape[0] == 1


def test_preprocess_data_negative_subscription_age():
    """Test preprocessing with negative subscription age."""
    df = pd.DataFrame({"id": [1], "subscription_age": [-1], "download_avg": [100]})
    processed_df, scaler = preprocess_data(df=df, return_scaler=True)
    assert processed_df["subscription_age"].iloc[0] >= 0
