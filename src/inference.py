
import pandas as pd

#функція для прогнозування
def predict_churn(model, data):
    return model.predict(data)

def preprocess_input(df=None):
    expected_columns = [
        'is_tv_subscriber', 'is_movie_package_subscriber',
        'subscription_age', 'reamining_contract',
        'service_failure_count', 'download_avg', 'upload_avg',
        'download_over_limit_0', 'download_over_limit_1', 'download_over_limit_2',
        'download_over_limit_3', 'download_over_limit_4', 'download_over_limit_5',
        'download_over_limit_6', 'download_over_limit_7'
    ]

    # Зробимо one-hot
    for i in range(8):
        df[f"download_over_limit_{i}"] = (df["download_over_limit"] == i).astype(int)

    # Видаляємо стару колонку
    df.drop(columns=["download_over_limit"], errors="ignore", inplace=True)

    # Забезпечимо наявність усіх колонок
    df = df.reindex(columns=expected_columns, fill_value=0)

    return df


