import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import logging
import pickle
import json
import os
import uuid
from typing import List, Dict
import numpy as np

logger = logging.getLogger(__name__)


def display_home_page(config: Dict) -> None:
    """Display the home page."""
    try:
        st.image(config["paths"]["logo"], width=150)
        st.title(
            config["ui"]["languages"][st.session_state.get("lang", "en")]["home_title"]
        )
        st.markdown(
            config["ui"]["languages"][st.session_state.get("lang", "en")][
                "home_description"
            ]
        )
        st.markdown("<br>", unsafe_allow_html=True)
        logger.info("Home page displayed successfully.")
    except FileNotFoundError as e:
        st.error(
            f"Logo file not found: {e}. Please ensure the logo file exists at the specified path. [Contact Support](#)"
        )
        logger.error(f"Logo file not found: {e}")
    except Exception as e:
        st.error(
            f"Error displaying home page: {e}. Please try refreshing the page or contact support. [Contact Support](#)"
        )
        logger.error(f"Error displaying home page: {e}")


def display_instructions(config: Dict) -> None:
    """Display instructions for CSV file with a checklist."""
    with st.expander(
        config["ui"]["languages"][st.session_state.get("lang", "en")][
            "instructions_label"
        ]
    ):
        st.markdown(
            config["ui"]["languages"][st.session_state.get("lang", "en")][
                "instructions_text"
            ].format(
                "\n".join(
                    [
                        f"- **{col}**: {config['ui']['column_descriptions'].get(col, 'Description missing')}"
                        for col in config["ui"]["required_columns"]
                    ]
                )
            )
        )
        st.markdown(
            """
        ### Checklist for CSV Upload
        Ensure your CSV file is correctly formatted to avoid errors during prediction:
        - [ ] **Ensure all required columns are present**: The model requires specific columns (listed above) to make \
        accurate predictions.
        - [ ] **Verify column names match exactly**: For example, use `remaining_contract`, not `reamining_contract`, \
        to avoid misinterpretation.
        - [ ] **Check that numeric columns contain valid numbers**: Non-numeric values will cause errors during \
         processing.
        - [ ] **Confirm the file is in CSV format with a valid separator**: Use a comma (`,`) or ensure the separator \
        is consistent.
        - [ ] **Follow the example structure**: Download the test data file below to see the required format.
        """
        )
        st.markdown("<br>", unsafe_allow_html=True)


def display_csv_template(config: Dict) -> None:
    """Display and provide a downloadable test CSV file."""
    dtypes = {
        "id": int,
        "is_tv_subscriber": int,
        "is_movie_package_subscriber": int,
        "subscription_age": float,
        "remaining_contract": float,
        "service_failure_count": int,
        "download_avg": float,
        "upload_avg": float,
        "download_over_limit": int,
    }
    st.write(
        "Test Data Example: This dataset includes three sample clients with varied attributes to demonstrate \
        the prediction functionality."
    )
    test_data = pd.DataFrame(
        {
            "id": pd.Series([1001, 1002, 1003], dtype=dtypes["id"]),
            **{
                col: pd.Series([0, 0, 0], dtype=dtypes[col])
                for col in config["ui"]["required_columns"]
            },
        }
    )
    test_data.iloc[:, 1:] = [
        [1, 0, 2.5, 1.0, 0, 100.0, 10.0, 0],
        [0, 1, 1.2, 0.0, 2, 50.0, 5.0, 3],
        [1, 1, 3.0, 0.5, 1, 75.0, 7.5, 1],
    ]
    st.dataframe(test_data)
    test_csv = test_data.to_csv(index=False)
    st.download_button(
        label=config["ui"]["languages"][st.session_state.get("lang", "en")][
            "download_test_data"
        ],
        data=test_csv,
        file_name="test_churn_prediction.csv",
        mime="text/csv",
    )
    st.markdown("<br>", unsafe_allow_html=True)


def display_single_prediction(pred: float, client_id: int, config: Dict) -> None:
    """Display gauge for single prediction."""
    high_risk_color = config["ui"]["colors"]["high_risk"]
    medium_risk_color = config["ui"]["colors"]["medium_risk"]
    low_risk_color = config["ui"]["colors"]["low_risk"]
    st.subheader(
        config["ui"]["languages"][st.session_state.get("lang", "en")][
            "single_prediction_title"
        ].format(client_id=client_id)
    )
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pred,
            title={
                "text": config["ui"]["languages"][st.session_state.get("lang", "en")][
                    "gauge_title"
                ]
            },
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {
                        "range": [0, config["ui"]["thresholds"]["low_risk"]],
                        "color": low_risk_color,
                    },
                    {
                        "range": [
                            config["ui"]["thresholds"]["low_risk"],
                            config["ui"]["thresholds"]["high_risk"],
                        ],
                        "color": medium_risk_color,
                    },
                    {
                        "range": [config["ui"]["thresholds"]["high_risk"], 1],
                        "color": high_risk_color,
                    },
                ],
            },
        )
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)


def display_multiple_predictions(
    predictions: List[float], client_ids: List[int], config: Dict
) -> None:
    """Display horizontal bar chart for multiple predictions."""
    high_risk_color = config["ui"]["colors"]["high_risk"]
    medium_risk_color = config["ui"]["colors"]["medium_risk"]
    low_risk_color = config["ui"]["colors"]["low_risk"]
    st.subheader(
        config["ui"]["languages"][st.session_state.get("lang", "en")][
            "multiple_predictions_title"
        ]
    )
    logger.info("Creating horizontal bar chart with Plotly...")
    fig = go.Figure(
        data=[
            go.Bar(
                y=[f"ID: {cid}" for cid in client_ids],
                x=predictions,
                orientation="h",
                marker_color=[
                    (
                        high_risk_color
                        if p > config["ui"]["thresholds"]["high_risk"]
                        else (
                            medium_risk_color
                            if p > config["ui"]["thresholds"]["low_risk"]
                            else low_risk_color
                        )
                    )
                    for p in predictions
                ],
                text=[f"{p:.2f}" for p in predictions],
                textposition="auto",
                hovertemplate="ID: %{y}<br>Probability: %{x:.2f}<br>Risk: %{customdata}<extra></extra>",
                customdata=[
                    (
                        config["ui"]["languages"][st.session_state.get("lang", "en")][
                            "high_risk_label"
                        ]
                        if p > config["ui"]["thresholds"]["high_risk"]
                        else (
                            config["ui"]["languages"][
                                st.session_state.get("lang", "en")
                            ]["medium_risk_label"]
                            if p > config["ui"]["thresholds"]["low_risk"]
                            else config["ui"]["languages"][
                                st.session_state.get("lang", "en")
                            ]["low_risk_label"]
                        )
                    )
                    for p in predictions
                ],
            )
        ]
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=config["ui"]["languages"][st.session_state.get("lang", "en")][
                "high_risk_label"
            ],
            marker=dict(color=high_risk_color),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=config["ui"]["languages"][st.session_state.get("lang", "en")][
                "medium_risk_label"
            ],
            marker=dict(color=medium_risk_color),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=config["ui"]["languages"][st.session_state.get("lang", "en")][
                "low_risk_label"
            ],
            marker=dict(color=low_risk_color),
        )
    )
    fig.update_layout(
        title=config["ui"]["languages"][st.session_state.get("lang", "en")][
            "multiple_predictions_title"
        ],
        xaxis_title=config["ui"]["languages"][st.session_state.get("lang", "en")][
            "xaxis_title"
        ],
        yaxis_title=config["ui"]["languages"][st.session_state.get("lang", "en")][
            "yaxis_title"
        ],
        height=min(800, max(400, 50 * len(predictions))),
        showlegend=True,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)


def display_results(predictions: List[float], config: Dict) -> None:
    """Display prediction results with visualizations and recommendations."""
    try:
        high_risk_color = config["ui"]["colors"]["high_risk"]
        medium_risk_color = config["ui"]["colors"]["medium_risk"]
        low_risk_color = config["ui"]["colors"]["low_risk"]
        st.markdown(
            """
            <style>
                .stApp { font-family: 'Arial', sans-serif; }
                h3, h4 { color: #333; margin-bottom: 10px; }
                .result-box {
                    padding: 10px; border-radius: 5px; margin-bottom: 10px;
                    color: white; font-weight: bold; width: 100%; box-sizing: border-box;
                }
                @media (max-width: 600px) {
                    .result-box { font-size: 12px; padding: 6px; }
                    .stPlotlyChart { height: 300px !important; }
                    .stDataFrame { font-size: 12px; }
                }
            </style>
        """,
            unsafe_allow_html=True,
        )

        st.subheader(
            config["ui"]["languages"][st.session_state.get("lang", "en")][
                "results_title"
            ]
        )
        recommendation = config["ui"]["languages"][st.session_state.get("lang", "en")][
            "recommendations"
        ]
        client_ids = st.session_state.get("original_ids")

        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()
        elif not isinstance(predictions, list):
            raise ValueError(
                f"Expected predictions to be a list, got {type(predictions)}"
            )

        if not client_ids or len(client_ids) != len(predictions):
            client_ids = list(range(1, len(predictions) + 1))

        logger.info(f"Client IDs: {client_ids[:5]}")
        logger.info(f"Predictions: {predictions[:5]}")

        with st.expander(
            config["ui"]["languages"][st.session_state.get("lang", "en")][
                "detailed_results_label"
            ],
            expanded=False,
        ):
            for idx, pred in enumerate(predictions):
                if not isinstance(pred, (int, float)):
                    raise ValueError(
                        f"Expected pred to be a float, got {type(pred)}: {pred}"
                    )
                risk_level = (
                    config["ui"]["languages"][st.session_state.get("lang", "en")][
                        "high_risk_label"
                    ]
                    if pred > config["ui"]["thresholds"]["high_risk"]
                    else (
                        config["ui"]["languages"][st.session_state.get("lang", "en")][
                            "medium_risk_label"
                        ]
                        if pred > config["ui"]["thresholds"]["low_risk"]
                        else config["ui"]["languages"][
                            st.session_state.get("lang", "en")
                        ]["low_risk_label"]
                    )
                )
                color = (
                    high_risk_color
                    if pred > config["ui"]["thresholds"]["high_risk"]
                    else (
                        medium_risk_color
                        if pred > config["ui"]["thresholds"]["low_risk"]
                        else low_risk_color
                    )
                )
                client_id = client_ids[idx]
                st.markdown(
                    f"""
                    <div class='result-box' style='background-color:{color};'>
                        ⚠️ {config["ui"]["languages"][st.session_state.get("lang", "en")]["client_label"]} \
                        (ID: {client_id}): {risk_level} {config["ui"]["languages"][st.session_state.get("lang", "en")]["probability_label"]} — {pred:.2f}
                    </div>
                    <div style='padding:5px; color:black;'>
                        {recommendation[risk_level].format(client_id=client_id)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if len(predictions) == 1:
            display_single_prediction(predictions[0], client_ids[0], config)
        else:
            display_multiple_predictions(predictions, client_ids, config)

        unique_key = f"results_table_{uuid.uuid4().hex}"
        logger.info(f"Displaying results table with unique key: {unique_key}")
        display_results_table(predictions, client_ids, config, unique_key=unique_key)
        logger.info("Prediction results displayed successfully.")
        st.markdown("<br>", unsafe_allow_html=True)
    except Exception as e:
        st.error(
            f"{config['ui']['languages'][st.session_state.get('lang', 'en')]['results_error']}: {e}. Please \
            check your input data and try again. [Contact Support](#)"
        )
        logger.error(f"Error displaying results: {e}")


def display_results_table(
    predictions: List[float], client_ids: List[int], config: Dict, unique_key: str
) -> None:
    """Display a table of prediction results with filters and a pie chart."""
    lang = st.session_state.get("lang", "en")
    logger.info(f"Rendering results table with key: {unique_key}")

    if not predictions or not client_ids:
        st.warning(
            "No predictions or client IDs available. Please make a prediction first. [Contact Support](#)"
        )
        logger.error("No predictions or client IDs available in session state.")
        return

    if isinstance(predictions, np.ndarray):
        predictions = predictions.tolist()

    risk_levels = [
        (
            "high"
            if p > config["ui"]["thresholds"]["high_risk"]
            else "medium" if p > config["ui"]["thresholds"]["low_risk"] else "low"
        )
        for p in predictions
    ]

    results_df = pd.DataFrame(
        {
            "client_id": client_ids,
            "probability": [f"{p:.2f}" for p in predictions],
            "risk_level_code": risk_levels,
        }
    )

    logger.info(
        f"Risk levels distribution: {results_df['risk_level_code'].value_counts().to_dict()}"
    )
    logger.info(f"Client IDs: {client_ids[:5]}")
    logger.info(f"Predictions: {predictions[:5]}")

    label_map = {
        "high": config["ui"]["languages"][lang]["high_risk_label"],
        "medium": config["ui"]["languages"][lang]["medium_risk_label"],
        "low": config["ui"]["languages"][lang]["low_risk_label"],
    }

    st.subheader("Filter Options")
    col1, col2 = st.columns(2)
    with col1:
        risk_filter_option = st.selectbox(
            config["ui"]["languages"][lang]["filter_label"],
            options=["all", "high", "medium", "low"],
            format_func=lambda x: {
                "all": config["ui"]["languages"][lang]["all_categories_label"],
                "high": config["ui"]["languages"][lang]["high_risk_label"],
                "medium": config["ui"]["languages"][lang]["medium_risk_label"],
                "low": config["ui"]["languages"][lang]["low_risk_label"],
            }[x],
            key=f"risk_filter_selectbox_{unique_key}",
        )
    with col2:
        probability_range = st.slider(
            "Churn Probability Range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.01,
            key=f"probability_filter_{unique_key}",
        )

    reset_key = f"reset_filters_{unique_key}"
    if st.button("Reset Filters", key=reset_key):
        st.session_state[f"risk_filter_selectbox_{unique_key}"] = "all"
        st.session_state[f"probability_filter_{unique_key}"] = (0.0, 1.0)
        st.session_state[f"client_id_filter_{unique_key}"] = ""
        st.experimental_rerun()

    logger.info(
        f"Selected filter option: {risk_filter_option}, Probability range: {probability_range}"
    )

    filtered_df = results_df.copy()
    if risk_filter_option != "all":
        filtered_df = filtered_df[filtered_df["risk_level_code"] == risk_filter_option]
    filtered_df = filtered_df[
        (filtered_df["probability"].astype(float) >= probability_range[0])
        & (filtered_df["probability"].astype(float) <= probability_range[1])
    ]

    client_id_filter = st.text_input(
        "Search by Client ID", key=f"client_id_filter_{unique_key}"
    )
    if client_id_filter:
        try:
            client_id_filter = int(client_id_filter)
            filtered_df = filtered_df[filtered_df["client_id"] == client_id_filter]
        except ValueError:
            st.warning("Please enter a valid Client ID (numeric).")
            client_id_filter = None

    st.subheader(config["ui"]["languages"][lang]["risk_distribution_label"])
    risk_counts = results_df["risk_level_code"].value_counts()
    fig = go.Figure(
        data=[
            go.Pie(
                labels=[label_map[level] for level in risk_counts.index],
                values=risk_counts.values,
                marker_colors=[
                    (
                        config["ui"]["colors"]["high_risk"]
                        if level == "high"
                        else (
                            config["ui"]["colors"]["medium_risk"]
                            if level == "medium"
                            else config["ui"]["colors"]["low_risk"]
                        )
                    )
                    for level in risk_counts.index
                ],
                textinfo="percent+label",
                hoverinfo="label+percent+value",
            )
        ]
    )
    fig.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig, use_container_width=True)

    risk_category_col = config["ui"]["languages"][lang]["risk_category_column"]
    results_df[risk_category_col] = results_df["risk_level_code"].map(label_map)
    filtered_df[risk_category_col] = filtered_df["risk_level_code"].map(label_map)

    st.subheader(config["ui"]["languages"][lang]["results_table_title"])
    st.write("Risk Distribution Summary:")
    st.write(results_df[risk_category_col].value_counts())

    if filtered_df.empty:
        st.warning(
            f"No clients found for selected filters: Risk Category = {label_map.get(risk_filter_option, 'All')}, "
            f"Probability Range = {probability_range}, Client ID = {client_id_filter or 'None'}"
        )
        logger.warning(
            f"No clients found for filters: {risk_filter_option}, {probability_range}, {client_id_filter}"
        )
    else:
        st.dataframe(
            filtered_df.drop(columns=["risk_level_code"]), use_container_width=True
        )

    csv = filtered_df.drop(columns=["risk_level_code"]).to_csv(index=False)
    st.download_button(
        label=config["ui"]["languages"][lang]["download_results_label"],
        data=csv,
        file_name="prediction_results.csv",
        mime="text/csv",
        key=f"download_results_button_{unique_key}",
    )
    st.markdown("<br>", unsafe_allow_html=True)


def display_model_page(config: Dict) -> None:
    """Display model information page."""
    try:
        st.title(
            config["ui"]["languages"][st.session_state.get("lang", "en")][
                "model_info_title"
            ]
        )
        st.markdown(
            config["ui"]["languages"][st.session_state.get("lang", "en")][
                "model_description"
            ]
        )

        metrics_path = config["paths"]["metrics"]
        if not os.path.exists(metrics_path):
            st.warning(
                f"Metrics file not found at {metrics_path}. Please ensure the file exists. [Contact Support](#)"
            )
            logger.error(f"Metrics file not found: {metrics_path}")
            return

        file_size = os.path.getsize(metrics_path)
        logger.info(f"Loading metrics file: {metrics_path}, size: {file_size} bytes")
        if file_size == 0:
            st.warning(
                f"Metrics file at {metrics_path} is empty. Please provide a valid metrics file. [Contact Support](#)"
            )
            logger.error(f"Metrics file is empty: {metrics_path}")
            return

        try:
            if metrics_path.endswith(".json"):
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                logger.info("Loaded metrics as JSON")
            elif metrics_path.endswith(".pkl"):
                with open(metrics_path, "rb") as f:
                    metrics = pickle.load(f)
                logger.info("Loaded metrics as pickle")
            else:
                st.error(
                    f"Unsupported file format for metrics: {metrics_path}. Expected .pkl or .json. [Contact Support](#)"
                )
                logger.error(f"Unsupported file format: {metrics_path}")
                return

            required_keys = ["accuracy", "0", "1"]
            missing_keys = [key for key in required_keys if key not in metrics]
            if missing_keys:
                st.error(
                    f"Metrics file is missing required keys: {missing_keys}. Please check the file structure. \
                    [Contact Support](#)"
                )
                logger.error(f"Missing keys in metrics: {missing_keys}")
                return

            st.subheader(
                config["ui"]["languages"][st.session_state.get("lang", "en")][
                    "metrics_title"
                ]
            )
            metrics_data = {
                "Metric": [
                    "Accuracy",
                    "Precision (0)",
                    "Recall (0)",
                    "F1-score (0)",
                    "Precision (1)",
                    "Recall (1)",
                    "F1-score (1)",
                ],
                "Value": [
                    metrics["accuracy"],
                    metrics["0"]["precision"],
                    metrics["0"]["recall"],
                    metrics["0"]["f1-score"],
                    metrics["1"]["precision"],
                    metrics["1"]["recall"],
                    metrics["1"]["f1-score"],
                ],
            }
            metrics_df = pd.DataFrame(metrics_data)
            st.table(metrics_df)

            if "roc_curve" in metrics:
                st.subheader("ROC Curve")
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=metrics["roc_curve"]["fpr"],
                        y=metrics["roc_curve"]["tpr"],
                        mode="lines",
                        name="ROC Curve",
                        line=dict(color="#1f77b4"),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Random Guess",
                        line=dict(color="gray", dash="dash"),
                    )
                )
                fig.update_layout(
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    template="plotly_white",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

            if "confusion_matrix" in metrics:
                st.subheader("Confusion Matrix")
                cm = metrics["confusion_matrix"]
                fig = go.Figure(
                    data=go.Heatmap(
                        z=cm,
                        x=["Predicted Negative", "Predicted Positive"],
                        y=["Actual Negative", "Actual Positive"],
                        colorscale="Blues",
                        text=cm,
                        texttemplate="%{text}",
                        showscale=True,
                    )
                )
                fig.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig, use_container_width=True)
        except (json.JSONDecodeError, pickle.UnpicklingError) as e:
            st.warning(
                f"Error deserializing metrics file: {e}. Please ensure the file is a \
                valid {'JSON' if metrics_path.endswith('.json') else 'pickle'} file. [Contact Support](#)"
            )
            logger.error(f"Error deserializing metrics file: {e}")
        except Exception as e:
            st.error(
                f"Error loading metrics: {e}. Please ensure the metrics file is correctly formatted. \
                [Contact Support](#)"
            )
            logger.error(f"Error loading metrics: {e}")

        try:
            with open(config["paths"]["model"], "rb") as f:
                model = pickle.load(f)
            feature_importance = pd.DataFrame(
                {
                    "feature": model.feature_names_in_,
                    "importance": model.feature_importances_,
                }
            ).sort_values(by="importance", ascending=False)
            st.subheader(
                config["ui"]["languages"][st.session_state.get("lang", "en")][
                    "feature_importance_caption"
                ]
            )
            fig = go.Figure(
                go.Bar(
                    x=feature_importance["importance"],
                    y=feature_importance["feature"],
                    orientation="h",
                    marker_color="#1f77b4",
                    text=feature_importance["importance"].round(3),
                    textposition="auto",
                )
            )
            fig.update_layout(
                xaxis_title=config["ui"]["languages"][
                    st.session_state.get("lang", "en")
                ]["xaxis_title"],
                yaxis_title="Feature",
                template="plotly_white",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)
        except FileNotFoundError:
            st.warning(
                f"Model file not found: {config['paths']['model']}. Please ensure the file exists. [Contact Support](#)"
            )
            logger.error(f"Model file not found: {config['paths']['model']}")
        except pickle.UnpicklingError as e:
            st.warning(
                f"Error deserializing model file: {e}. Please check the model file format. [Contact Support](#)"
            )
            logger.error(f"Error deserializing model file: {e}")

        logger.info("Model page displayed successfully.")
        st.markdown("<br>", unsafe_allow_html=True)
    except Exception as e:
        st.error(
            f"{config['ui']['languages'][st.session_state.get('lang', 'en')]['model_page_error']}: {e}. Please \
            try refreshing the page or contact support. [Contact Support](#)"
        )
        logger.error(f"Error displaying model page: {e}")
