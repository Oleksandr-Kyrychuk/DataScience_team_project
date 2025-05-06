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

## Вимоги
- Python 3.12 (обов’язково для сумісності).
- Встановлені залежності з `requirements.txt`.

## Налаштування оточення
1. Переконайтеся, що у вас встановлено Python 3.12:
Якщо немає, завантажте з [python.org](https://www.python.org/downloads/) і встановіть.
2. Створіть віртуальне оточення: py -3.12 -m venv .venv
3. Активуйте оточення: .\venv\Scripts\Activate.ps1 (Для Linux/Mac: `source .venv/bin/activate`)
4. Встановіть залежності: pip install -r requirements.txt
5. Налаштуйте pre-commit для автоматичних перевірок: pre-commit install


## Процес роботи
1. Створюйте нову гілку для кожної задачі:

git checkout -b feature/<назва-задачі>

Приклад: `git checkout -b feature/eda-analysis`.
2. Додайте Змінені файли git add .(або точний файл)
3. Робіть коміти:
- Перед комітом запускайте `pre-commit run --all-files`, щоб уникнути помилок (наприклад, `end-of-file-fixer`).
- Виконуйте коміт: `git commit -m "Опис змін"`.
4. Пуште зміни:

git push origin feature/<назва-задачі>
text
4. Створюйте Pull Request (PR) на GitHub:
- Назва: “Опис задачі” (наприклад, “Add EDA notebook”).
- Опис: Короткий опис змін і призначення ревьювера.

5. Синхронізуйте локальну гілку `main`:

git checkout main
git pull origin main
