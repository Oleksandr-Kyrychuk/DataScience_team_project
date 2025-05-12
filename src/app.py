#імпортування фреймворків  
import streamlit as st
import pandas as pd
import joblib
from inference import predict_churn, preprocess_input

#завантаження моделі 

model = joblib.load("model.pkl")
data = None 

#створення стрімліт застосунку
st.title("Прогнозування Відтоку Клієнтів для Телекомунікаційної компанії")

input_type = st.radio('Оберіть вид завантаження даних клієнта:', ['Завантажити дані у форматі CSV', 'Ввести вручну'])

if input_type == 'Завантажити дані у форматі CSV':
    st.subheader("Завантаження даних клієнта файлом")
    uploaded_file = st.file_uploader("Оберіть CSV файл", type="csv")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("Файл успішно завантажено!")
            st.dataframe(data)
        except Exception as e:
            st.error(f"Не вдалося прочитати файл: {e}")
    else:
        st.info("Очікую на файл...")
        
elif input_type == 'Ввести вручну':
    id = st.number_input('Введіть id')
    is_tv_subscriber = int(st.checkbox('Є підписка на ТБ'))
    is_movie_package_subscriber = int(st.checkbox('Є підписка на пакет з фільмами'))
    subscription_age = st.number_input('Введіть вік підписки')
    bill_avg = st.number_input('Введіть середній чек')
    reamining_contract= int(st.checkbox('Є reamining_contract'))
    service_failure_count = st.number_input('Введіть кількість відмов сервісу')
    download_avg = st.number_input('Введіть середнє завантаження')
    upload_avg = st.number_input('Введіть середнє вивантаження')
    download_over_limit = st.number_input('Скачування поза лімітом')

    data = pd.DataFrame([{
        "id":id,
        "is_tv_subscriber":is_tv_subscriber,
        "is_movie_package_subscriber":is_movie_package_subscriber,
        "subscription_age":subscription_age,
        "bill_avg":bill_avg,
        "reamining_contract":reamining_contract,
        "service_failure_count":service_failure_count,
        "download_avg":download_avg,
        "upload_avg":upload_avg,
        "download_over_limit":download_over_limit,
    }])


if data is not None:
    if st.button("Зробити прогноз"):
        processed_data = preprocess_input(data)
        preds = predict_churn(model, processed_data)

        st.subheader("Результати прогнозу:")
        for i, p in enumerate(preds):
            st.write(f"Клієнт {i+1}: Ймовірність відтоку — {p}")
