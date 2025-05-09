import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys


def preprocess_data(data_path=None, df=None):
    """
    Препроцесинг даних для моделі відтоку клієнтів.

    Args:
        data_path (str, optional): Шлях до CSV файлу з даними.
        df (pd.DataFrame, optional): Вхідний DataFrame, якщо дані вже завантажені.

    Returns:
        pd.DataFrame: Оброблений DataFrame, готовий для моделювання.
    """
    # Завантаження даних
    if data_path is not None:
        df = pd.read_csv(data_path)
    elif df is None:
        raise ValueError("Потрібно вказати data_path або df")

    df_churn = df.copy()

    # Обробка пропусків
    df_churn["reamining_contract"] = df_churn["reamining_contract"].fillna(0)
    df_churn["download_avg"] = df_churn["download_avg"].fillna(df_churn["download_avg"].median())
    df_churn["upload_avg"] = df_churn["upload_avg"].fillna(df_churn["upload_avg"].median())

    # Заміна негативних значень subscription_age на медіану
    if (df_churn["subscription_age"] < 0).any():
        median_age = df_churn.loc[df_churn["subscription_age"] >= 0, "subscription_age"].median()
        df_churn.loc[df_churn["subscription_age"] < 0, "subscription_age"] = median_age

    # Обмеження викидів у download_avg за допомогою IQR
    Q1 = df_churn["download_avg"].quantile(0.25)
    Q3 = df_churn["download_avg"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df_churn["download_avg"] = np.where(
        df_churn["download_avg"] > upper,
        upper,
        np.where(df_churn["download_avg"] < lower, lower, df_churn["download_avg"]),
    )

    # Обмеження викидів у upload_avg за допомогою IQR
    Q1 = df_churn["upload_avg"].quantile(0.25)
    Q3 = df_churn["upload_avg"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df_churn["upload_avg"] = np.where(
        df_churn["upload_avg"] > upper,
        upper,
        np.where(df_churn["upload_avg"] < lower, lower, df_churn["upload_avg"]),
    )

    # Видалення екстремальних значень bill_avg
    bill_upper = df_churn["bill_avg"].quantile(0.99)
    df_churn = df_churn[df_churn["bill_avg"] <= bill_upper]

    # One-Hot Encoding
    download_dummies = pd.get_dummies(df_churn["download_over_limit"], prefix="download_over_limit")
    df_churn = pd.concat([df_churn.drop(columns=["download_over_limit"]), download_dummies], axis=1)

    # Нормалізація числових ознак
    scaler = StandardScaler()
    numeric_cols = [
        "subscription_age",
        "bill_avg",
        "reamining_contract",
        "service_failure_count",
        "download_avg",
        "upload_avg",
    ]
    df_churn[numeric_cols] = scaler.fit_transform(df_churn[numeric_cols])

    # Видалення id та bill_avg (слабка кореляція)
    df_churn.drop(columns=["id", "bill_avg"], inplace=True)

    return df_churn


if __name__ == "__main__":
    # Тестування локально з аргументом командного рядка або за замовчуванням
    data_path = "data/internet_service_churn.csv"  # За замовчуванням
    if len(sys.argv) > 1:
        data_path = sys.argv[1]  # Можна передати шлях через командний рядок
    try:
        processed_df = preprocess_data(data_path=data_path)
        print("Перші 5 рядків після препроцесингу:")
        print(processed_df.head())
        print("\nРозмір датасету:", processed_df.shape)
    except FileNotFoundError:
        print("Файл не знайдено. Перевір шлях до data_path.")
