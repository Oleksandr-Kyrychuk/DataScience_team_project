import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
import os
from typing import Tuple, Optional, Union


def preprocess_data(
    data_path: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    return_scaler: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, StandardScaler]]:
    """Preprocess data for the churn model."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_data_path = os.path.join(base_dir, "datasets", "internet_service_churn.csv")

    if df is not None:
        df_churn = df.copy()
    else:
        if data_path is None:
            data_path = default_data_path
        if os.path.exists(data_path):
            df_churn = pd.read_csv(data_path)
        else:
            raise FileNotFoundError(
                f"File not found at path: {data_path}. Check path or provide df."
            )

    # Rename reamining_contract to remaining_contract
    if (
        "reamining_contract" in df_churn.columns
        and "remaining_contract" not in df_churn.columns
    ):
        df_churn.rename(
            columns={"reamining_contract": "remaining_contract"}, inplace=True
        )
        print("Renamed column 'reamining_contract' to 'remaining_contract'.")

    df_churn["remaining_contract"] = df_churn["remaining_contract"].fillna(0)
    df_churn["download_avg"] = df_churn["download_avg"].fillna(
        df_churn["download_avg"].median()
    )
    df_churn["upload_avg"] = df_churn["upload_avg"].fillna(
        df_churn["upload_avg"].median()
    )

    if (df_churn["subscription_age"] < 0).any():
        median_age = df_churn.loc[
            df_churn["subscription_age"] >= 0, "subscription_age"
        ].median()
        df_churn.loc[df_churn["subscription_age"] < 0, "subscription_age"] = median_age

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

    bill_upper = df_churn["bill_avg"].quantile(0.99)
    df_churn = df_churn[df_churn["bill_avg"] <= bill_upper]

    download_dummies = pd.get_dummies(
        df_churn["download_over_limit"], prefix="download_over_limit"
    )
    df_churn = pd.concat(
        [df_churn.drop(columns=["download_over_limit"]), download_dummies], axis=1
    )

    scaler = StandardScaler()
    numeric_cols = [
        "subscription_age",
        "remaining_contract",  # Змінено з "reamining_contract"
        "service_failure_count",
        "download_avg",
        "upload_avg",
    ]
    df_churn[numeric_cols] = scaler.fit_transform(df_churn[numeric_cols])

    df_churn.drop(columns=["id", "bill_avg"], inplace=True)

    if return_scaler:
        return df_churn, scaler
    return df_churn


if __name__ == "__main__":
    data_path = None
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    try:
        processed_df = preprocess_data(data_path=data_path, return_scaler=False)
        print("First 5 rows after preprocessing:")
        print(processed_df.head())
        print("\nDataset size:", processed_df.shape)
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Error: {e}")
