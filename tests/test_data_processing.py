import pytest
import pandas as pd
from src.data_processing import load_csv_data, validate_csv_dtypes
from src.session_manager import load_config


@pytest.fixture
def config():
    return load_config()


def test_load_csv_data_missing_columns(config):
    df = pd.DataFrame({"id": [1], "subscription_age": [2.5]})
    with pytest.raises(SystemExit):  # st.stop() викликає SystemExit у Streamlit
        load_csv_data(config)


def test_validate_csv_dtypes_non_numeric(config):
    df = pd.DataFrame({"is_tv_subscriber": ["invalid"], "subscription_age": [2.5]})
    with pytest.raises(SystemExit):
        validate_csv_dtypes(df, config)
