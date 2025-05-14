import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from inference import predict_churn, preprocess_input
import logging
import os
from streamlit_option_menu import option_menu
import streamlit_lottie
import json
from matplotlib.patches import Patch
import numpy as np

if "a" not in st.session_state:
    st.session_state.a = np.array([])

# Налаштування логування
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Додавання CSS для стилізації логотипу
st.markdown(
    """
<style>
    .logo-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-bottom: 20px;
    }
    .logo-img {
        width: 200px;
        padding: 10px;
    }
    .stButton>button {
        background-color: #F63366;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        padding: 10px;
        margin-top: 20px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Шлях до кореня проєкту
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logo_path = os.path.join(project_root, "assets", "logo black.svg")


def load_model_and_scaler(model_path, scaler_path):
    """Завантажує модель і scaler із заданих шляхів."""
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        logger.info("Модель і scaler успішно завантажено з кешу.")
        return model, scaler
    except Exception as e:
        st.error(f"Не вдалося завантажити модель або scaler: {e}")
        logger.error(f"Не вдалося завантажити модель або scaler: {e}")
        st.stop()


def display_home_page():
    """Відображає вміст вкладки 'Головна'."""
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    try:
        st.image(logo_path, use_container_width=False, output_format="SVG")
    except Exception as e:
        st.error(f"Помилка завантаження логотипу: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<h1 style='text-align: center; color: #262730;'>ChurnVision:\
         Зменшуйте відтік клієнтів</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: gray;'>Прогнозуйте \
        ризики відтоку за секунди. Без коду.</p>",
        unsafe_allow_html=True,
    )
    if st.button("🚀 Спробувати зараз", key="try_now", use_container_width=True):
        st.session_state.selected_tab = "Прогноз"
        st.rerun()
    st.markdown(
        """
    ### Чому ChurnVision?
    - **Швидко**: Аналітика за секунди.
    - **Просто**: Завантажте CSV або введіть дані.
    - **Ефективно**: Зменшуйте відтік клієнтів на 20%.
    """,
        unsafe_allow_html=True,
    )


def load_csv_data():
    """Завантажує дані з CSV-файлу та оновлює стан сесії."""
    uploaded_file = st.file_uploader("Оберіть CSV файл", type="csv", key="csv_uploader")
    if uploaded_file is not None:
        try:
            st.session_state.data = pd.read_csv(uploaded_file)
            id_col = None
            for col in st.session_state.data.columns:
                if col.lower() in ["id", "client_id", "customer_id"]:
                    id_col = col
                    break
            if id_col:
                st.session_state.original_ids = st.session_state.data[id_col].copy()
                logger.info(f"Оригінальні ID збережено з стовпця: {id_col}")
            else:
                st.session_state.original_ids = None
                logger.warning("Стовпець 'id' не знайдено в CSV. Використовуються автоматичні ID.")
                st.warning(
                    "Стовпець 'id' не знайдено в CSV. ID клієнтів будуть згенеровані автоматично."
                )

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
            missing_cols = [
                col for col in required_cols if col not in st.session_state.data.columns
            ]
            if missing_cols:
                st.error(
                    f"Увага! Відсутні необхідні колонки: {missing_cols}. "
                    "Вони будуть заповнені нулями, що може суттєво вплинути на точність прогнозу. "
                    "Будь ласка, перевірте ваш датасет або скористайтеся шаблоном CSV."
                )
                logger.warning(f"Відсутні колонки при завантаженні CSV: {missing_cols}")

            st.success("Файл успішно завантажено!")
            logger.info("Файл CSV успішно завантажено.")
            st.dataframe(st.session_state.data)
            # Оновлюємо input_type після успішного завантаження CSV
            st.session_state.input_type = "Завантажити дані у форматі CSV"
        except Exception as e:
            st.error(f"Не вдалося прочитати файл: {e}")
            logger.error(f"Не вдалося прочитати файл: {e}")
            st.session_state.data = None
            st.session_state.original_ids = None
    else:
        if st.session_state.input_type != "Ввести вручну":
            st.session_state.data = None
            st.session_state.original_ids = None
        st.info("Очікую на файл...")


def load_manual_data():
    """Завантажує дані, введені вручну, та виконує прогноз."""
    with st.form("client_form"):
        id = st.number_input("Введіть id", min_value=0, step=1)
        is_tv_subscriber = int(st.checkbox("Є підписка на ТБ"))
        is_movie_package_subscriber = int(st.checkbox("Є підписка на пакет з фільмами"))
        subscription_age = st.number_input("Введіть вік підписки (роки)", min_value=0.0, step=0.1)
        reamining_contract = st.number_input(
            "Введіть залишок контракту (роки)", min_value=0.0, step=0.1
        )
        service_failure_count = st.number_input(
            "Введіть кількість відмов сервісу", min_value=0, step=1
        )
        download_avg = st.number_input("Введіть середнє завантаження (МБ)", min_value=0.0, step=0.1)
        upload_avg = st.number_input("Введіть середнє вивантаження (МБ)", min_value=0.0, step=0.1)
        download_over_limit = st.selectbox(
            "Скачування поза лімітом", options=[0, 1, 2, 3, 4, 5, 6, 7]
        )

        submitted = st.form_submit_button("Зробити прогноз")

        if submitted:
            if subscription_age < 0 or download_avg < 0 or upload_avg < 0:
                st.error("Введені значення не можуть бути від’ємними!")
                logger.error("Введені від’ємні значення при ручному вводі.")
            else:
                st.session_state.data = pd.DataFrame(
                    [
                        {
                            "id": id,
                            "is_tv_subscriber": is_tv_subscriber,
                            "is_movie_package_subscriber": is_movie_package_subscriber,
                            "subscription_age": subscription_age,
                            "reamining_contract": reamining_contract,
                            "service_failure_count": service_failure_count,
                            "download_avg": download_avg,
                            "upload_avg": upload_avg,
                            "download_over_limit": download_over_limit,
                        }
                    ]
                )
                st.session_state.original_ids = [id]
                logger.info("Дані успішно створено з ручного введення.")
                st.session_state.input_type = "Ввести вручну"

                try:
                    processed_data = preprocess_input(
                        st.session_state.data, scaler=scaler, logger=logger
                    )
                    preds = predict_churn(model, processed_data, logger=logger)
                    logger.info(
                        f"Прогноз виконано для ручного введення. Кількість клієнтів: {len(preds)}"
                    )
                    st.session_state.preds = preds
                    st.session_state.show_results = True
                except Exception as e:
                    st.error(f"Помилка під час прогнозування: {str(e)}")
                    logger.error(f"Помилка під час прогнозування: {str(e)}")
                    st.session_state.show_results = False


def make_csv_prediction():
    """Виконує прогноз для завантажених CSV-даних."""
    if st.session_state.data is not None:
        try:
            processed_data = preprocess_input(st.session_state.data, scaler=scaler, logger=logger)
            preds = predict_churn(model, processed_data, logger=logger)
            logger.info(f"Прогноз виконано для CSV. Кількість клієнтів: {len(preds)}")
            st.session_state.preds = preds
            st.session_state.show_results = True
            # Оновлюємо st.session_state.a для збереження прогнозів
            st.session_state.a = np.array(preds)
        except Exception as e:
            st.error(f"Помилка під час прогнозування: {str(e)}")
            logger.error(f"Помилка під час прогнозування: {str(e)}")
            st.session_state.show_results = False
    else:
        st.error("Будь ласка, завантажте CSV-файл перед прогнозуванням.")
        st.session_state.show_results = False


def display_results(preds, input_type):
    """Відображає результати прогнозу з фільтрацією та візуалізацією."""
    # Фільтр за рівнем ризику
    risk_filter = st.selectbox(
        "Фільтрувати за рівнем ризику", ["Усі", "Висока", "Середня", "Низька"]
    )
    filtered_preds = []
    filtered_ids = []
    for i, p in enumerate(preds):
        level = "Висока" if p > 0.7 else "Середня" if p >= 0.3 else "Низька"
        if risk_filter == "Усі" or risk_filter == level:
            filtered_preds.append(p)
            filtered_ids.append(
                st.session_state.original_ids[i]
                if st.session_state.original_ids is not None
                else i + 1
            )

    logger.info(
        f"Filtered preds: {filtered_preds}, Filtered IDs: {filtered_ids}"
    )  # Для відлагодження

    # Анімація після прогнозу
    def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)

    try:
        lottie_animation = load_lottiefile(
            os.path.join(project_root, "assets", "success_animation.json")
        )
        streamlit_lottie.st_lottie(lottie_animation, height=100)
    except Exception as e:
        logger.warning(f"Не вдалося завантажити анімацію: {e}")

    # Загальна статистика для CSV
    if input_type == "Завантажити дані у форматі CSV" and len(preds) > 0:  # Виправлена умова
        high_risk = sum(1 for p in preds if p > 0.7)
        medium_risk = sum(1 for p in preds if 0.3 <= p <= 0.7)
        low_risk = sum(1 for p in preds if p < 0.3)
        avg_churn_prob = sum(preds) / len(preds)

        st.markdown("### Загальна статистика")
        st.markdown(
            f"""
        - **Клієнтів із високим ризиком відтоку (>0.7)**: {high_risk}
        - **Клієнтів із середнім ризиком (0.3-0.7)**: {medium_risk}
        - **Клієнтів із низьким ризиком (<0.3)**: {low_risk}
        - **Середня ймовірність відтоку**: {avg_churn_prob:.2f}
        """
        )

    # Відображення відфільтрованих результатів
    st.subheader("Результати прогнозу:")
    recommendation = {
        "Висока": "Рекомендується \
        зв’язатися з клієнтом (ID: {client_id})\
         для пропозиції знижок або інших утримуючих заходів.",
        "Середня": "Клієнт (ID: {client_id}) може бути в зоні ризику, варто звернути увагу.",
        "Низька": "Клієнт (ID: {client_id}), ймовірно, залишиться з компанією.",
    }
    if not filtered_preds:
        st.warning("Немає даних для відображення після фільтрації. Спробуйте змінити фільтр.")
        return

    for i, p in enumerate(filtered_preds):
        client_id = filtered_ids[i]
        if p > 0.7:
            color = "red"
            level = "Висока"
        elif p < 0.3:
            color = "green"
            level = "Низька"
        else:
            color = "orange"
            level = "Середня"

        st.markdown(
            f"""
            <div style='background-color:{color}; \
            padding:10px; border-radius:5px; color:white; font-weight:bold; margin-bottom:10px'>
                ⚠️ Клієнт (ID: {client_id}): {level} ймовірність відтоку — {p:.2f}
            </div>
            <div style='padding:5px; color:black;'>
                {recommendation[level].format(client_id=client_id)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Візуалізація для одного користувача
    if len(filtered_preds) == 1:
        client_id = filtered_ids[0]
        st.subheader(f"Візуалізація ймовірності відтоку для клієнта (ID: {client_id})")
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=filtered_preds[0],
                title={"text": "Ймовірність відтоку"},
                gauge={
                    "axis": {"range": [0, 1]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 0.3], "color": "green"},
                        {"range": [0.3, 0.7], "color": "orange"},
                        {"range": [0.7, 1], "color": "red"},
                    ],
                },
            )
        )
        st.plotly_chart(fig)

    # Таблиця та експорт для CSV
    if input_type == "Завантажити дані у форматі CSV" and len(filtered_preds) > 0:
        results_df = pd.DataFrame(
            {
                "ID клієнта": filtered_ids,
                "Ймовірність відтоку": [f"{p:.2f}" for p in filtered_preds],
                "Категорія ризику": [
                    "Висока" if p > 0.7 else "Середня" if p > 0.3 else "Низька"
                    for p in filtered_preds
                ],
            }
        )
        st.subheader("Підсумкова таблиця результатів")
        st.dataframe(results_df)
        results_csv = results_df.to_csv(index=False)
        st.download_button(
            label="Завантажити результати прогнозу",
            data=results_csv,
            file_name="churn_predictions.csv",
            mime="text/csv",
        )

    # Візуалізація для кількох клієнтів
    if input_type == "Завантажити дані у форматі CSV" and len(filtered_preds) > 1:
        st.subheader("Візуалізація ймовірностей відтоку")
        logger.info(f"Створюємо гістограму для {len(filtered_preds)} клієнтів...")
        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(
            range(len(filtered_preds)),
            filtered_preds,
            color=["red" if p > 0.7 else "orange" if p > 0.3 else "green" for p in filtered_preds],
        )
        ax.set_xticks(range(len(filtered_preds)))
        ax.set_xticklabels(filtered_ids, rotation=90, ha="center")
        for bar, p in zip(bars, filtered_preds):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
            )
        ax.grid(True, axis="y", linestyle="--", alpha=0.7)
        legend_elements = [
            Patch(facecolor="red", label="Висока ймовірність (>0.7)"),
            Patch(facecolor="orange", label="Середня ймовірність (0.3-0.7)"),
            Patch(facecolor="green", label="Низька ймовірність (<0.3)"),
        ]
        ax.legend(handles=legend_elements, loc="best")
        ax.set_xlabel("Клієнт (ID)")
        ax.set_ylabel("Ймовірність відтоку")
        ax.set_title("Ймовірності відтоку клієнтів")
        st.pyplot(fig)
        plt.clf()
    else:
        logger.info(
            f"Гістограма не відображається: input_type={input_type},\
             len(filtered_preds)={len(filtered_preds)}"
        )


def display_model_page():
    """Відображає вміст вкладки 'Про модель'."""
    st.markdown("## Про модель ChurnVision")
    st.markdown("Тут буде інформація про модель, метрики та важливість ознак.")


# Ініціалізація стану для вибору вкладки
if "selected_tab" not in st.session_state:
    st.session_state.selected_tab = "Головна"

# Меню
with st.sidebar:
    selected = option_menu(
        "ChurnVision",
        ["Головна", "Прогноз", "Про модель"],
        icons=["house", "graph-up", "info-circle"],
        default_index=["Головна", "Прогноз", "Про модель"].index(st.session_state.selected_tab),
    )

# Оновлення стану вкладки при виборі
st.session_state.selected_tab = selected

# Основна логіка додатку
if selected == "Головна":
    display_home_page()
elif selected == "Прогноз":
    model, scaler = load_model_and_scaler(
        os.path.join(project_root, "model.pkl"), os.path.join(project_root, "scaler.pkl")
    )
    if "data" not in st.session_state:
        st.session_state.data = None
    if "input_type" not in st.session_state:
        st.session_state.input_type = "Завантажити дані у форматі CSV"
    if "original_ids" not in st.session_state:
        st.session_state.original_ids = None
    if "show_results" not in st.session_state:
        st.session_state.show_results = False
    if "preds" not in st.session_state:
        st.session_state.preds = None

    # Документація про необхідні колонки
    with st.expander("📋 Інструкція для CSV-файлу"):
        st.markdown(
            """
            Для коректного прогнозування ваш CSV-файл повинен містити наступні колонки:
            - **id**: Унікальний ідентифікатор клієнта (ціле число, наприклад, 1001)
            - **is_tv_subscriber**: Чи є підписка на ТБ (0 або 1)
            - **is_movie_package_subscriber**: Чи є підписка на пакет фільмів (0 або 1)
            - **subscription_age**: Вік підписки в роках (додатне число)
            - **reamining_contract**:\
             Залишок контракту в роках (додатне число або 0)
            - **service_failure_count**: Кількість відмов сервісу (ціле число ≥ 0)
            - **download_avg**: Середнє завантаження\
             в МБ (додатне число)
            - **upload_avg**: Середнє вивантаження в МБ (додатне число)
            - **download_over_limit**:\
             Кількість перевищень ліміту завантаження (ціле число від 0 до 7)

            Якщо будь-яка з колонок відсутня, вона буде заповнена нулями, \
            що може вплинути на точність прогнозу.
            """
        )

    # Шаблон CSV
    template_data = pd.DataFrame(
        {
            "id": [1001],
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
    template_csv = template_data.to_csv(index=False)
    st.download_button(
        label="Завантажити шаблон CSV",
        data=template_csv,
        file_name="template_churn_prediction.csv",
        mime="text/csv",
    )

    # Кнопка "Очистити"
    if st.button("🗑️ Очистити", key="clear_button"):
        st.session_state.data = None
        st.session_state.original_ids = None
        st.session_state.preds = None
        st.session_state.show_results = False
        st.session_state.input_type = "Завантажити дані у форматі CSV"
        st.rerun()

    # Вкладки для введення
    tabs = st.tabs(["📄 Завантажити CSV", "✍️ Ввести вручну"])
    with tabs[0]:
        st.subheader("Завантаження даних клієнта файлом")
        load_csv_data()
        if st.button("Зробити прогноз для CSV", key="predict_csv"):
            make_csv_prediction()
    with tabs[1]:
        st.subheader("Введення даних клієнта")
        load_manual_data()

    # Відображення результатів
    if "preds" in st.session_state and st.session_state.show_results:
        display_results(st.session_state.preds, st.session_state.input_type)

# Очищення стану при переході на "Головна"
if selected == "Головна":
    if "show_results" in st.session_state:
        st.session_state.show_results = False
    if "preds" in st.session_state:
        st.session_state.preds = None
    if "data" in st.session_state:
        st.session_state.data = None
    if "original_ids" in st.session_state:
        st.session_state.original_ids = None

# Повідомлення про відсутність результатів
if selected != "Прогноз" and (
    "show_results" not in st.session_state or not st.session_state.show_results
):
    st.info("Будь ласка, введіть або завантажте дані та зробіть прогноз на вкладці 'Прогноз'.")
