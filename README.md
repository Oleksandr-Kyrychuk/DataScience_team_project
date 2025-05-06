# DataScience_team_project: Customer Churn Prediction

Проєкт із прогнозування відтоку клієнтів для компанії.

## Структура проєкту
- `data/` - Датасети (зберігаються в Google Drive)
- `notebooks/` - Jupyter-ноутбуки для аналізу
  - `eda.ipynb` - Початковий аналіз (Анна)
- `src/` - Код
  - `preprocessing.py` - Обробка даних (Богдан)
  - `app.py` - Streamlit-інтерфейс (Григорій)
- `Dockerfile` - Контейнер для розгортання (Григорій)
- `requirements.txt` - Залежності

## Як почати
1. Клонуйте: `git clone git@github.com:Oleksandr-Kyrychuk/DataScience_team_project.git`
2. Створіть гілку: `git checkout -b feature/ваша-назва`
3. Завантажте датасет: [dataset.csv](посилання_на_Google_Drive)
4. Встановіть залежності: `pip install -r requirements.txt`
5. Запустіть Streamlit (поки в розробці): `streamlit run src/app.py`

## Команда
- Анна: EDA (`notebooks/eda.ipynb`)
- Богдан: Обробка даних (`src/preprocessing.py`)
- Вікторія: Модель (`src/model.py`)
- Григорій: Streamlit і Docker (`src/app.py`, `Dockerfile`)