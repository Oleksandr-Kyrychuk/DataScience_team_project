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
        bill_avg = st.number_input("Введіть середній чек", min_value=0.0, step=1.0)
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
            if subscription_age < 0 or download_avg < 0 or upload_avg < 0 or bill_avg < 0:
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
                            "bill_avg": bill_avg,
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

            # Візуалізація результатів
            if len(preds) > 1:
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
