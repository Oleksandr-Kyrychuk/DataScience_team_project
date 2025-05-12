import pandas as pd
import numpy as np


def preprocess_input(df=None, scaler=None):
    """
    Препроцесинг вхідних даних для передбачення.

    Args:
        df (pd.DataFrame): Вхідний DataFrame.
        scaler (StandardScaler): Об'єкт StandardScaler для нормалізації.

    Returns:
        pd.DataFrame: Оброблений DataFrame, готовий для передбачення.
    """
    if df is None or df.empty:
        raise ValueError("Вхідний DataFrame не може бути порожнім або None.")

    expected_columns = [
        "is_tv_subscriber",
        "is_movie_package_subscriber",
        "subscription_age",
        "reamining_contract",
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

    # Перевірка наявності необхідних колонок
    required_cols = [
        "is_tv_subscriber",
        "is_movie_package_subscriber",
        "subscription_age",
        "reamining_contract",
        "service_failure_count",
        "download_avg",
        "upload_avg",
        "download_over_limit",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Відсутні необхідні колонки: {missing_cols}")

    # Обробка пропусків
    df["reamining_contract"] = df["reamining_contract"].fillna(0)
    df["download_avg"] = df["download_avg"].fillna(df["download_avg"].median())
    df["upload_avg"] = df["upload_avg"].fillna(df["upload_avg"].median())

    # Заміна негативних значень subscription_age на медіану
    if (df["subscription_age"] < 0).any():
        median_age = df.loc[df["subscription_age"] >= 0, "subscription_age"].median()
        df.loc[df["subscription_age"] < 0, "subscription_age"] = median_age

    # Обмеження викидів у download_avg за допомогою IQR
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

    # Обмеження викидів у upload_avg за допомогою IQR
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

    # One-Hot Encoding для download_over_limit
    download_dummies = pd.get_dummies(df["download_over_limit"], prefix="download_over_limit")
    df = pd.concat(
        [df.drop(columns=["download_over_limit"], errors="ignore"), download_dummies], axis=1
    )

    # Нормалізація (без bill_avg, яке видаляється через низьку кореляцію)
    numeric_cols = [
        "subscription_age",
        "reamining_contract",
        "service_failure_count",
        "download_avg",
        "upload_avg",
    ]
    if scaler is not None:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    else:
        raise ValueError("Scaler is required for preprocessing.")

    # Видаляємо зайві колонки, включаючи bill_avg (низька кореляція, див. звіт)
    df.drop(columns=["id", "bill_avg"], errors="ignore", inplace=True)

    # Забезпечуємо наявність усіх колонок
    df = df.reindex(columns=expected_columns, fill_value=0)

    return df


def predict_churn(model, data):
    """
    Прогнозування ймовірності відтоку.

    Args:
        model: Навчена модель.
        data (pd.DataFrame): Оброблені дані.

    Returns:
        np.ndarray: Ймовірності відтоку (клас 1).
    """
    return model.predict_proba(data)[:, 1]
