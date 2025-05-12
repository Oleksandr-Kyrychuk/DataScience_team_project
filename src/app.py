import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from inference import predict_churn, preprocess_input
import logging

# Налаштування логування
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Завантаження моделі та scaler
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    logger.info("Модель і scaler успішно завантажено.")
except Exception as e:
    st.error(f"Не вдалося завантажити модель або scaler: {e}")
    logger.error(f"Не вдалося завантажити модель або scaler: {e}")
    st.stop()

# Створення Streamlit-додатка
st.title("Прогнозування Відтоку Клієнтів для Телекомунікаційної компанії")

# Документація про необхідні колонки
st.markdown(
    """
### Інструкція для CSV-файлу
Для коректного прогнозування ваш CSV-файл повинен містити наступні колонки:
- **is_tv_subscriber**: Чи є підписка на ТБ (0 або 1)
- **is_movie_package_subscriber**: Чи є підписка на пакет фільмів (0 або 1)
- **subscription_age**: Вік підписки в роках (додатне число)
- **reamining_contract**: Залишок контракту в роках (додатне число або 0)
- **service_failure_count**: Кількість відмов сервісу (ціле число ≥ 0)
- **download_avg**: Середнє завантаження в МБ (додатне число)
- **upload_avg**: Середнє вивантаження в МБ (додатне число)
- **download_over_limit**: Кількість перевищень ліміту завантаження (ціле число від 0 до 7)

Якщо будь-яка з колонок відсутня, вона буде заповнена нулями, що може вплинути на точність прогнозу.
"""
)

# Шаблон CSV для завантаження
template_data = pd.DataFrame(
    {
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

input_type = st.radio(
    "Оберіть вид завантаження даних клієнта:", ["Завантажити дані у форматі CSV", "Ввести вручну"]
)

data = None

if input_type == "Завантажити дані у форматі CSV":
    st.subheader("Завантаження даних клієнта файлом")
    uploaded_file = st.file_uploader("Оберіть CSV файл", type="csv")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
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
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                st.error(
                    f"Увага! Відсутні необхідні колонки: {missing_cols}. "
                    "Вони будуть заповнені нулями, що може суттєво вплинути на точність прогнозу. "
                    "Будь ласка, перевірте ваш датасет або скористайтеся шаблоном CSV."
                )
                logger.warning(f"Відсутні колонки при завантаженні CSV: {missing_cols}")
            st.success("Файл успішно завантажено!")
            logger.info("Файл CSV успішно завантажено.")
            st.dataframe(data)
        except Exception as e:
            st.error(f"Не вдалося прочитати файл: {e}")
            logger.error(f"Не вдалося прочитати файл: {e}")
    else:
        st.info("Очікую на файл...")

elif input_type == "Ввести вручну":
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
                data = pd.DataFrame(
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
                logger.info("Дані успішно створено з ручного введення.")

if data is not None:
    if st.button("Зробити прогноз") or (input_type == "Ввести вручну" and data is not None):
        try:
            processed_data = preprocess_input(data, scaler=scaler, logger=logger)
            preds = predict_churn(model, processed_data, logger=logger)

            st.subheader("Результати прогнозу:")
            for i, p in enumerate(preds):
                if p > 0.7:
                    st.write(f"Клієнт {i+1}: **Висока ймовірність відтоку** — {p:.2f}")
                elif p < 0.3:
                    st.write(f"Клієнт {i+1}: **Низька ймовірність відтоку** — {p:.2f}")
                else:
                    st.write(f"Клієнт {i+1}: **Середня ймовірність відтоку** — {p:.2f}")

            # Візуалізація результатів лише для CSV (більше одного клієнта)
            if input_type == "Завантажити дані у форматі CSV" and len(preds) > 1:
                st.subheader("Візуалізація ймовірностей відтоку")
                fig, ax = plt.subplots()
                ax.bar(
                    range(1, len(preds) + 1),
                    preds,
                    color=["red" if p > 0.7 else "orange" if p > 0.3 else "green" for p in preds],
                )
                ax.set_xlabel("Клієнт")
                ax.set_ylabel("Ймовірність відтоку")
                ax.set_title("Ймовірності відтоку клієнтів")
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Помилка під час прогнозування: {str(e)}")
            logger.error(f"Помилка під час прогнозування: {str(e)}")
else:
    st.info("Будь ласка, введіть або завантажте дані для прогнозування.")
