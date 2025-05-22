import pytest
import pandas as pd
from src.inference import preprocess_input, predict_churn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def test_preprocess_input_empty_dataframe():
    """Test preprocessing with empty DataFrame."""
    df = pd.DataFrame()
    scaler = StandardScaler()
    with pytest.raises(ValueError, match="DataFrame is empty"):
        preprocess_input(df, scaler)


def test_predict_churn_valid_input():
    """Test prediction with valid input."""
    df = pd.DataFrame(
        {
            "is_tv_subscriber": [1],
            "is_movie_package_subscriber": [0],
            "subscription_age": [2.5],
            "reamining_contract": [1.0],
            "service_failure_count": [0],
            "download_avg": [100.0],
            "upload_avg": [10.0],
            "download_over_limit": [0],
        }
    )
    scaler = StandardScaler()
    model = RandomForestClassifier()
    scaler.fit(df)  # Fit scaler for test
    processed_data = preprocess_input(df, scaler)
    predictions = predict_churn(model, processed_data)
    assert len(predictions) == 1
    assert 0 <= predictions[0] <= 1
