import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from inference import predict_churn, preprocess_input
import logging
import os

# Налаштування логування
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Визначаємо корінь проєкту
current_dir = os.path.dirname(os.path.abspath(__file__))  # Поточна директорія (src)
project_root = os.path.dirname(current_dir)  # Корінь проєкту (на один рівень вище)

# Завантаження моделі та scaler із кореня проєкту
try:
    model_path = os.path.join(project_root, "model.pkl")
    scaler_path = os.path.join(project_root, "scaler.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    logger.info("Модель і scaler успішно завантажено.")
except Exception as e:
    st.error(f"Не вдалося завантажити модель або scaler: {e}")
    logger.error(f"Не вдалося завантажити модель або scaler: {e}")
    st.stop()

# Ініціалізація стану сесії для збереження даних
if "data" not in st.session_state:
    st.session_state.data = None
if "input_type" not in st.session_state:
    st.session_state.input_type = "Завантажити дані у форматі CSV"
if "original_ids" not in st.session_state:
    st.session_state.original_ids = None

# Створення Streamlit-додатка
st.title("Прогнозування Відтоку Клієнтів для Телекомунікаційної компанії")

# Документація про необхідні колонки (випадаючий список)
with st.expander("📋 Інструкція для CSV-файлу"):
    st.markdown(
        """
        Для коректного прогнозування ваш CSV-файл повинен містити наступні колонки:
        - **id**: Унікальний ідентифікатор клієнта (ціле число, наприклад, 1001)
        - **is_tv_subscriber**: Чи є підписка на ТБ (0 або 1)
        - **is_movie_package_subscriber**: Чи є підписка на пакет фільмів (0 або 1)
        - **subscription_age**: Вік підписки в роках (додатне число)
        - **reamining_contract**: Залишок контракту в роках (додатне число або 0)
        - **service_failure_count**: Кількість відмов сервісу (ціле число ≥ 0)
        - **download_avg**: Середнє завантаження в МБ (додатне число)
        - **upload_avg**: Середнє вивантаження в МБ (додатне число)
        - **download_over_limit**: Кількість перевищень ліміту завантаження (ціле число від 0 до 7)

        Якщо будь-яка з колонок відсутня, вона буде заповнена нулями, \
        що може вплинути на точність прогнозу.
        """
    )

# Шаблон CSV для завантаження (додаємо стовпець id)
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

# Вкладки для вибору режиму введення
tabs = st.tabs(["📄 Завантажити CSV", "✍️ Ввести вручну"])

# Вкладка 1: Завантаження CSV
with tabs[0]:
    st.subheader("Завантаження даних клієнта файлом")
    uploaded_file = st.file_uploader("Оберіть CSV файл", type="csv", key="csv_uploader")

    if uploaded_file is not None:
        try:
            st.session_state.data = pd.read_csv(uploaded_file)
            # Зберігаємо оригінальні ID перед обробкою
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

            # Перевірка колонок перед обробкою
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

    st.session_state.input_type = "Завантажити дані у форматі CSV"

    # Додаємо кнопку "Зробити прогноз" у вкладці CSV
    if st.button("Зробити прогноз для CSV", key="predict_csv"):
        if st.session_state.data is not None:
            try:
                processed_data = preprocess_input(
                    st.session_state.data, scaler=scaler, logger=logger
                )
                preds = predict_churn(model, processed_data, logger=logger)
                logger.info(f"Прогноз виконано для CSV. Кількість клієнтів: {len(preds)}")

                st.session_state.preds = preds
                st.session_state.show_results = True
            except Exception as e:
                st.error(f"Помилка під час прогнозування: {str(e)}")
                logger.error(f"Помилка під час прогнозування: {str(e)}")
                st.session_state.show_results = False
        else:
            st.error("Будь ласка, завантажте CSV-файл перед прогнозуванням.")
            st.session_state.show_results = False

# Вкладка 2: Ручне введення
with tabs[1]:
    st.subheader("Введення даних клієнта")
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
            # Валідація введених даних
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
                st.session_state.original_ids = [id]  # Зберігаємо введений ID
                logger.info("Дані успішно створено з ручного введення.")
                st.session_state.input_type = "Ввести вручну"

                # Виконуємо прогноз для ручного введення
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

# Відображення результатів прогнозування
if (
    "show_results" in st.session_state
    and st.session_state.show_results
    and "preds" in st.session_state
):
    preds = st.session_state.preds
    input_type = st.session_state.input_type

    st.subheader("Результати прогнозу:")
    recommendation = {
        "Висока": "Рекомендується зв’язатися з клієнтом (ID: {client_id}) для пропозиції знижок\
         або інших утримуючих заходів.",
        "Середня": "Клієнт (ID: {client_id}) може бути в зоні ризику, варто звернути увагу.",
        "Низька": "Клієнт (ID: {client_id}), ймовірно, залишиться з компанією.",
    }
    for i, p in enumerate(preds):
        # Отримуємо ID клієнта перед умовними операторами
        client_id = (
            st.session_state.original_ids[i] if st.session_state.original_ids is not None else i + 1
        )

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
            <div style='
                background-color:{color};
                padding:10px;
                border-radius:5px;
                color:white;
                font-weight:bold;
                margin-bottom:10px'>
                ⚠️ Клієнт (ID: {client_id}): {level} ймовірність відтоку — {p:.2f}
            </div>
            <div style='padding:5px; color:black;'>
                {recommendation[level].format(client_id=client_id)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Візуалізація для одного користувача
    if len(preds) == 1:
        client_id = (
            st.session_state.original_ids[0] if st.session_state.original_ids is not None else 1
        )
        st.subheader(f"Візуалізація ймовірності відтоку для клієнта (ID: {client_id})")
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=preds[0],
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
    if input_type == "Завантажити дані у форматі CSV":
        # Використовуємо збережені оригінальні ID
        if st.session_state.original_ids is not None:
            client_ids = st.session_state.original_ids
        else:
            client_ids = range(1, len(preds) + 1)

        results_df = pd.DataFrame(
            {
                "ID клієнта": client_ids,
                "Ймовірність відтоку": [f"{p:.2f}" for p in preds],
                "Категорія ризику": [
                    "Висока" if p > 0.7 else "Середня" if p > 0.3 else "Низька" for p in preds
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

    # Візуалізація для кількох клієнтів (датасет)
    if input_type == "Завантажити дані у форматі CSV" and len(preds) > 1:
        st.subheader("Візуалізація ймовірностей відтоку")
        logger.info("Створюємо гістограму для датасету...")
        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(10, 6))

        # Використовуємо оригінальні ID для осі X
        if st.session_state.original_ids is not None:
            x_labels = st.session_state.original_ids
        else:
            x_labels = range(1, len(preds) + 1)

        bars = ax.bar(
            range(len(preds)),
            preds,
            color=["red" if p > 0.7 else "orange" if p > 0.3 else "green" for p in preds],
        )
        ax.set_xticks(range(len(preds)))
        ax.set_xticklabels(x_labels, rotation=90, ha="center")

        for bar, p in zip(bars, preds):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
            )

        ax.grid(True, axis="y", linestyle="--", alpha=0.7)
        from matplotlib.patches import Patch

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
        plt.clf()  # Очищаємо графік після відображення
    else:
        logger.info(
            f"Гістограма не відображається: input_type={input_type}, len(preds)={len(preds)}"
        )
else:
    if "show_results" not in st.session_state or not st.session_state.show_results:
        st.info("Будь ласка, введіть або завантажте дані та зробіть прогноз.")
