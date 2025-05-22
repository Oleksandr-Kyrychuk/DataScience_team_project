import streamlit as st
import pandas as pd
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def validate_csv_dtypes(df: pd.DataFrame, config: Dict) -> None:
    """Validate data types of required columns in the DataFrame."""
    dtypes = {
        "is_tv_subscriber": int,
        "is_movie_package_subscriber": int,
        "subscription_age": float,
        "remaining_contract": float,
        "service_failure_count": int,
        "download_avg": float,
        "upload_avg": float,
        "download_over_limit": int,
    }
    for col, expected_type in dtypes.items():
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            logger.error(f"Column {col} must be of type {expected_type.__name__}")
            st.error(f"Column {col} must be numeric.")
            st.stop()
        if col in [
            "subscription_age",
            "download_avg",
            "upload_avg",
            "remaining_contract",
        ]:
            if (df[col] < 0).any():
                logger.error(f"Column {col} contains negative values")
                st.error(f"Column {col} cannot contain negative values.")
                st.stop()


def load_csv_data(config: Dict) -> Optional[pd.DataFrame]:
    try:
        uploaded_file = st.file_uploader(
            config["ui"]["languages"][st.session_state.get("lang", "en")][
                "upload_csv_label"
            ],
            type=["csv"],
        )
        if uploaded_file is not None:
            uploaded_file.seek(0)
            content = uploaded_file.read().decode("utf-8", errors="ignore")
            uploaded_file.seek(0)
            logger.info(f"Full CSV content (first 500 characters):\n{content[:500]}")
            lines = content.strip().split("\n")
            logger.info(f"Number of lines in CSV: {len(lines)}")
            if not content.strip():
                st.error("Uploaded CSV file is empty or contains only whitespace.")
                logger.error("Uploaded CSV file is empty or contains only whitespace.")
                st.stop()
            if len(lines) <= 1:
                st.error("CSV file contains only headers or is empty.")
                logger.error("CSV file contains only headers or is empty.")
                st.stop()
            if uploaded_file.size == 0:
                st.error("Uploaded CSV file is empty.")
                logger.error("Uploaded CSV file is empty.")
                st.stop()

            # Визначення сепаратора
            import csv

            sniffer = csv.Sniffer()
            try:
                dialect = sniffer.sniff(content[:1024])
                sep = dialect.delimiter
                logger.info(f"Detected CSV separator: '{sep}'")
            except csv.Error:
                sep = ","
                logger.warning("Could not detect CSV delimiter, defaulting to comma.")

            # Перевірка формату рядків
            header_cols = lines[0].split(sep)
            for i, line in enumerate(lines[1:], 1):
                values = line.split(sep)
                if len(values) != len(header_cols):
                    logger.warning(
                        f"Line {i+1} has {len(values)} columns, expected {len(header_cols)}: {line}"
                    )
                    st.error(f"Line {i+1} in CSV has incorrect number of columns.")
                    st.stop()

            # Спроба різних кодувань
            encodings = ["utf-8", "latin1", "cp1252"]
            df = None
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    logger.info(
                        f"Attempting to read CSV with encoding {encoding} and separator '{sep}'"
                    )
                    progress_bar = st.progress(0)
                    chunks = pd.read_csv(
                        uploaded_file, chunksize=1000, sep=sep, encoding=encoding
                    )
                    df_list = []
                    uploaded_file.seek(0)
                    total_chunks = sum(
                        1
                        for _ in pd.read_csv(
                            uploaded_file, chunksize=1000, sep=sep, encoding=encoding
                        )
                    )
                    uploaded_file.seek(0)
                    for i, chunk in enumerate(
                        pd.read_csv(
                            uploaded_file, chunksize=1000, sep=sep, encoding=encoding
                        )
                    ):
                        df_list.append(chunk)
                        progress_bar.progress((i + 1) / total_chunks)
                    df = pd.concat(df_list)
                    progress_bar.empty()
                    break
                except UnicodeDecodeError:
                    logger.warning(
                        f"Failed to decode with {encoding}, trying next encoding."
                    )
                    continue
                except pd.errors.EmptyDataError as e:
                    st.error(f"Empty CSV file with encoding {encoding}: {e}")
                    logger.error(f"Empty CSV file with encoding {encoding}: {e}")
                    st.stop()
            else:
                st.error("Could not decode CSV file with any supported encoding.")
                logger.error("Could not decode CSV file with any supported encoding.")
                st.stop()

            # Debug: Display columns
            st.write("Columns in CSV file:", df.columns.tolist())

            # Rename remaining_contract if necessary
            if (
                "reamining_contract" in df.columns
                and "remaining_contract" not in df.columns
            ):
                df.rename(
                    columns={"reamining_contract": "remaining_contract"}, inplace=True
                )
                logger.warning(
                    "Renamed column 'reamining_contract' to 'remaining_contract'."
                )

            required_cols = config["ui"]["required_columns"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                logger.error(f"Missing required columns: {', '.join(missing_cols)}")
                st.stop()

            validate_csv_dtypes(df, config)
            st.session_state["csv_data"] = df
            id_col = None
            for col in df.columns:
                if col.strip().lower() in ["id", "client_id", "customer_id"]:
                    id_col = col
                    break
            if id_col:
                st.session_state["original_ids"] = df[id_col].tolist()
                logger.info(f"Original IDs saved from column: {id_col}")
            else:
                st.session_state["original_ids"] = list(range(1, len(df) + 1))
                logger.warning("ID column not found, generated IDs 1, 2, 3...")
            st.write(
                config["ui"]["languages"][st.session_state.get("lang", "en")][
                    "data_preview"
                ]
            )
            st.dataframe(df.head())
            logger.info("CSV data loaded successfully.")
            return df
        return None
    except pd.errors.EmptyDataError as e:
        st.error(
            config["ui"]["languages"][st.session_state.get("lang", "en")][
                "empty_csv_error"
            ]
        )
        logger.error(f"Empty CSV file: {e}")
        st.stop()
    except Exception as e:
        st.error(
            f"{config['ui']['languages'][st.session_state.get('lang', 'en')]['csv_error']}: {e}"
        )
        logger.error(f"Error loading CSV file: {e}")
        st.stop()


def load_manual_data(config: Dict) -> pd.DataFrame:
    """Load manually entered data."""
    try:
        st.subheader(
            config["ui"]["languages"][st.session_state.get("lang", "en")][
                "manual_input_title"
            ]
        )
        data = {}
        for col in config["ui"]["required_columns"]:
            desc = config["ui"]["column_descriptions"].get(col, col)
            if col in ["is_tv_subscriber", "is_movie_package_subscriber"]:
                data[col] = st.selectbox(f"{col} ({desc})", [0, 1], key=col)
            elif col == "download_over_limit":
                data[col] = st.number_input(
                    f"{col} ({desc})", min_value=0, max_value=7, step=1, key=col
                )
            elif col == "service_failure_count":
                data[col] = st.number_input(
                    f"{col} ({desc})", min_value=0, step=1, key=col
                )
            else:
                data[col] = st.number_input(
                    f"{col} ({desc})", min_value=0.0, step=0.1, key=col
                )
        df = pd.DataFrame([data])
        st.session_state["manual_data"] = df
        st.session_state["original_ids"] = [1]
        logger.info("Manual data loaded successfully.")
        return df
    except Exception as e:
        st.error(
            f"{config['ui']['languages'][st.session_state.get('lang', 'en')]['manual_error']}: {e}"
        )
        logger.error(f"Error loading manual data: {e}")
        st.stop()
