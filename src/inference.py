import pandas as pd
import numpy as np
import logging


def preprocess_input(df=None, scaler=None, logger=None):
    """
    Препроцесинг вхідних даних для передбачення.

    Args:
        df (pd.DataFrame): Вхідний DataFrame.
        scaler (StandardScaler): Об'єкт StandardScaler для нормалізації.
        logger (logging.Logger, optional): Логер для запису повідомлень.

    Returns:
        pd.DataFrame: Оброблений DataFrame, готовий для передбачення.

    Raises:
        ValueError: Якщо дані некоректні або відсутні.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Перевірка на порожній або None DataFrame
    if df is None or df.empty:
        logger.error("Вхідний DataFrame не може бути порожнім або None.")
        raise ValueError("Вхідний DataFrame не може бути порожнім або None.")

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
        logger.error(f"Відсутні необхідні колонки: {missing_cols}")
        raise ValueError(f"Відсутні необхідні колонки: {missing_cols}")

    # Валідація типу даних для download_over_limit
    if not pd.api.types.is_numeric_dtype(df["download_over_limit"]):
        logger.error("Колонка 'download_over_limit' містить нечислові значення.")
        raise ValueError("Колонка 'download_over_limit' повинна містити числові значення.")

    # Конвертація до числового типу та обробка некоректних значень
    df["download_over_limit"] = pd.to_numeric(df["download_over_limit"], errors="coerce")
    df["download_over_limit"] = df["download_over_limit"].fillna(0).astype(int)
    # Обмеження значень до діапазону 0-7
    df["download_over_limit"] = df["download_over_limit"].clip(0, 7)

    # Обробка пропусків
    df["reamining_contract"] = df["reamining_contract"].fillna(0)
    df["download_avg"] = df["download_avg"].fillna(df["download_avg"].median())
    df["upload_avg"] = df["upload_avg"].fillna(df["upload_avg"].median())

    # Заміна негативних значень subscription_age на медіану
    if (df["subscription_age"] < 0).any():
        logger.warning("Знайдено від'ємні значення в 'subscription_age'. Замінюємо на медіану.")
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
    for i in range(8):
        df[f"download_over_limit_{i}"] = (df["download_over_limit"] == i).astype(int)
    df.drop(columns=["download_over_limit"], errors="ignore", inplace=True)

    # Нормалізація
    numeric_cols = [
        "subscription_age",
        "reamining_contract",
        "service_failure_count",
        "download_avg",
        "upload_avg",
    ]
    if scaler is None:
        logger.error("Scaler is required for preprocessing.")
        raise ValueError("Scaler is required for preprocessing.")
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Видаляємо зайві колонки
    df.drop(columns=["id", "bill_avg"], errors="ignore", inplace=True)

    # Забезпечуємо правильний порядок колонок
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
    df = df.reindex(columns=expected_columns, fill_value=0)
    logger.info("Дані успішно оброблені для передбачення.")
    return df


def predict_churn(model, data, logger=None):
    """
    Прогнозування ймовірності відтоку.

    Args:
        model: Навчена модель.
        data (pd.DataFrame): Оброблені дані.
        logger (logging.Logger, optional): Логер для запису повідомлень.

    Returns:
        np.ndarray: Ймовірності відтоку (клас 1).
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if model is None:
        logger.error("Модель не може бути None.")
        raise ValueError("Модель не може бути None.")

    if data is None or data.empty:
        logger.error("Вхідні дані не можуть бути порожніми або None.")
        raise ValueError("Вхідні дані не можуть бути порожніми або None.")

    try:
        predictions = model.predict_proba(data)[:, 1]
        logger.info("Передбачення успішно виконано.")
        return predictions
    except Exception as e:
        logger.error(f"Помилка під час передбачення: {str(e)}")
        raise ValueError(f"Помилка під час передбачення: {str(e)}")
