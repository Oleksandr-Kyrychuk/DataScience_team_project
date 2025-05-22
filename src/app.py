import streamlit as st
from streamlit_option_menu import option_menu
from session_manager import init_session_state, load_config
from ui_components import (
    display_home_page,
    display_results,
    display_model_page,
    display_instructions,
    display_csv_template,
)
from data_processing import load_csv_data, load_manual_data
from prediction import make_prediction
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Churn Prediction", layout="wide")

    # Load configuration and initialize session state
    config = load_config()
    init_session_state()

    # Debug: Log session state
    logger.info(f"Initial session state: {st.session_state}")

    # Theme selection
    theme = st.sidebar.selectbox("Theme", ["Light", "Dark"], key="theme")
    if theme == "Dark":
        st.markdown(
            """
            <style>
                .stApp { background-color: #1E1E1E; color: #FFFFFF; }
                h1, h2, h3, h4 { color: #FFFFFF; }
            </style>
        """,
            unsafe_allow_html=True,
        )

    # Sidebar menu
    with st.sidebar:
        lang = st.sidebar.selectbox(
            "Language", ["English", "Ukrainian"], key="lang_select"
        )
        st.session_state["lang"] = "en" if lang == "English" else "uk"
        selected = option_menu(
            "ChurnVision",
            [
                config["ui"]["languages"][st.session_state.get("lang", "en")][
                    "home_title"
                ],
                config["ui"]["languages"][st.session_state.get("lang", "en")][
                    "predict_title"
                ],
                config["ui"]["languages"][st.session_state.get("lang", "en")][
                    "model_info_title"
                ],
            ],
            icons=["house", "graph-up", "info-circle"],
            default_index=0,
        )

    # Map selected menu to internal keys
    menu_mapping = {
        config["ui"]["languages"][st.session_state.get("lang", "en")][
            "home_title"
        ]: "Home",
        config["ui"]["languages"][st.session_state.get("lang", "en")][
            "predict_title"
        ]: "Predict",
        config["ui"]["languages"][st.session_state.get("lang", "en")][
            "model_info_title"
        ]: "Model Info",
    }
    menu = menu_mapping.get(selected, "Home")

    if menu == "Home":
        display_home_page(config)

    elif menu == "Predict":

        st.title(
            config["ui"]["languages"][st.session_state.get("lang", "en")][
                "predict_title"
            ]
        )

        if not st.session_state.get("confirm_clear", False):
            if st.button(
                config["ui"]["languages"][st.session_state.get("lang", "en")][
                    "clear_button"
                ],
                key="clear_button_top",
            ):
                st.session_state["confirm_clear"] = True

        if st.session_state.get("confirm_clear", False):
            st.warning(
                config["ui"]["languages"][st.session_state.get("lang", "en")][
                    "clear_confirmation"
                ]
            )
            if st.button(
                config["ui"]["languages"][st.session_state.get("lang", "en")][
                    "confirm_clear_button"
                ],
                key="confirm_clear_button",
            ):
                keys_to_clear = [
                    "csv_data",
                    "manual_data",
                    "predictions",
                    "original_ids",
                ]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                init_session_state()
                st.session_state["cleared"] = True
                st.session_state["confirm_clear"] = False
                st.success(
                    config["ui"]["languages"][st.session_state.get("lang", "en")][
                        "clear_success"
                    ]
                )
                st.rerun()

        # Reset cleared flag after displaying results

        if "cleared" in st.session_state:
            del st.session_state["cleared"]

        tab1, tab2 = st.tabs(
            [
                config["ui"]["languages"][st.session_state.get("lang", "en")][
                    "upload_csv_tab"
                ],
                config["ui"]["languages"][st.session_state.get("lang", "en")][
                    "manual_input_tab"
                ],
            ]
        )

        with tab1:
            display_instructions(config)
            display_csv_template(config)
            df = load_csv_data(config)
            if df is not None:
                st.success(
                    f"{config['ui']['languages'][st.session_state.get('lang', 'en')]['csv_success']} {len(df)}"
                )
                if st.button(
                    config["ui"]["languages"][st.session_state.get("lang", "en")][
                        "predict_button_csv"
                    ],
                    key="predict_button_csv",
                ):
                    with st.spinner(
                        config["ui"]["languages"][st.session_state.get("lang", "en")][
                            "predicting"
                        ]
                    ):
                        predictions = make_prediction(df, config, data_source="csv")
                        st.session_state["predictions"] = predictions
                        logger.info(f"Predictions saved: {predictions[:5]}")
                        logger.info(
                            f"Session state after prediction: {st.session_state}"
                        )
                        display_results(predictions, config)
                # Показати результати, якщо вони є в сесії та не є None
                elif (
                    "predictions" in st.session_state
                    and st.session_state["predictions"] is not None
                ):
                    display_results(st.session_state["predictions"], config)

        with tab2:
            df = load_manual_data(config)
            if st.button(
                config["ui"]["languages"][st.session_state.get("lang", "en")][
                    "predict_button_manual"
                ],
                key="predict_button_manual",
            ):
                with st.spinner(
                    config["ui"]["languages"][st.session_state.get("lang", "en")][
                        "predicting"
                    ]
                ):
                    predictions = make_prediction(df, config, data_source="manual")
                    st.session_state["predictions"] = predictions
                    logger.info(f"Predictions saved: {predictions[:5]}")
                    logger.info(f"Session state after prediction: {st.session_state}")
                    display_results(predictions, config)
            # Показати результати, якщо вони є в сесії та не є None
            elif (
                "predictions" in st.session_state
                and st.session_state["predictions"] is not None
            ):
                display_results(st.session_state["predictions"], config)

    elif menu == "Model Info":
        display_model_page(config)


if __name__ == "__main__":
    main()
