# DataScience_team_project: Прогнозування відтоку клієнтів

Цей проєкт розроблено для прогнозування ймовірності відтоку клієнтів телекомунікаційної компанії на основі їхніх даних. Використовуючи модель RandomForest, проєкт ідентифікує клієнтів із високим ризиком припинення співпраці та пропонує рекомендації для їх утримання. Інтерфейс Streamlit дозволяє завантажувати CSV-файли або вводити дані вручну, а результати відображаються з візуалізаціями (гістограми, індикатори). Проєкт упаковано в Docker для легкого розгортання.

## Опис даних

1. Датасет `internet_service_churn.csv` – CSV-файл із даними клієнтів телекомунікаційної компанії.
2. Ознаки:
   - `id`: Унікальний ідентифікатор користувача.
   - `is_tv_subscriber`: Підписка на ТБ-послугу (0 або 1).
   - `is_movie_package_subscriber`: Підписка на пакет із фільмами (0 або 1).
   - `subscription_age`: Тривалість підписки в місяцях.
   - `bill_avg`: Середній щомісячний рахунок.
   - `reamining_contract`: Залишок контракту в місяцях (21,572 пропусків).
   - `service_failure_count`: Кількість скарг/проблем зі зв’язком.
   - `download_avg`: Середня швидкість завантаження (381 пропусків).
   - `upload_avg`: Середня швидкість відвантаження (381 пропусків).
   - `download_over_limit`: Перевищення ліміту трафіку (0 або 1).
   - `churn`: Відтік клієнта (0 – ні, 1 – так, 55.41% позитивного класу).
3. Джерело: Локально в `datasets/` або завантажте з [dataset.csv](посилання_на_Google_Drive) (шаблон доступний у Streamlit-додатку).

## Основні можливості

1. Прогнозування ймовірності відтоку клієнтів за допомогою RandomForest.
2. Інтерактивний Streamlit-інтерфейс для завантаження CSV або ручного введення даних.
3. Візуалізація результатів: гістограми для кількох клієнтів, індикатори для одного.
4. Контейнеризація через Docker і Docker Compose.

## Структура проєкту

1. `datasets/internet_service_churn.csv` – Датасет для прогнозування.
2. `docs/` – Документація та зображення (скріншоти, діаграми).
3. `notebooks/` – Jupyter-ноутбуки для аналізу:
   - `eda.ipynb` – Початковий аналіз.
4. `src/` – Основний код:
   - `preprocessing.py` – Обробка даних.
   - `app.py` – Streamlit-інтерфейс.
   - `inference.py` – Логіка прогнозування.
   - `model.py` – Навчання моделі.
5. `Dockerfile` – Конфігурація Docker-образу.
6. `docker-compose.yml` – Налаштування Docker Compose.
7. `requirements.txt` – Список залежностей.
8. `README.md` – Опис проєкту.

## Вимоги

1. Python 3.12 (рекомендується для сумісності, мінімальна версія 3.10 для Docker).
2. Docker і Docker Compose для контейнеризації.
3. Встановлені залежності з `requirements.txt`.
4. Датасет у `datasets/internet_service_churn.csv` або завантажений з [dataset.csv](посилання_на_Google_Drive).

## Як почати

1. Клонуйте репозиторій:
   ```bash
   git clone git@github.com:Oleksandr-Kyrychuk/DataScience_team_project.git
   ```
2. Створіть гілку:
   ```bash
   git checkout -b feature/ваша-назва
   ```
3. Завантажте датасет:
   - Переконайтеся, що файл доступний у `datasets/internet_service_churn.csv`.
   - Якщо ні, завантажте з [dataset.csv](посилання_на_Google_Drive) або використовуйте шаблон CSV із Streamlit-додатка.
4. Встановіть залежності:
   ```bash
   pip install -r requirements.txt
   ```
5. Запустіть Streamlit-додаток:
   ```bash
   streamlit run src/app.py
   ```

## Налаштування оточення

1. Переконайтеся, що встановлено Python 3.12 (або 3.10 для сумісності з Docker). Якщо ні, завантажте з [python.org](https://www.python.org/downloads/) і встановіть.
2. Створіть віртуальне оточення:
   ```bash
   py -3.12 -m venv .venv  # Windows
   python3.10 -m venv .venv  # Linux/Mac
   ```
3. Активуйте оточення:
   ```bash
   .\.venv\Scripts\Activate.ps1  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```
4. Встановіть залежності:
   ```bash
   pip install -r requirements.txt
   ```
5. Налаштуйте `pre-commit` для автоматичних перевірок:
   ```bash
   pre-commit install
   ```

## Запуск із Docker

1. Переконайтеся, що Docker і Docker Compose встановлені.
2. Запустіть проєкт:
   ```bash
   docker-compose up --build
   ```
3. Відкрийте `http://localhost:8501` у браузері.
4. Зупиніть контейнер:
   ```bash
   docker-compose down
   ```

## Контейнеризація

1. `Dockerfile` створює образ на базі Python 3.10, копіює код проєкту, встановлює залежності з `requirements.txt` і запускає `src/model.py` для підготовки моделі, а потім `src/app.py` для Streamlit на порту 8501.
2. `docker-compose.yml` автоматизує запуск контейнера однією командою `docker-compose up --build`, налаштовуючи мережу та порт 8501.
3. Контейнеризація забезпечує відтворюваність і легке розгортання на будь-якій системі з Docker.

## Процес роботи

1. Створюйте нову гілку для кожної задачі:
   ```bash
   git checkout -b feature/назва-задачі
   ```
   Приклад:
   ```bash
   git checkout -b feature/eda-analysis
   ```
2. Додавайте змінені файли:
   ```bash
   git add .  # або конкретний файл
   ```
3. Робіть коміти:
   - Перед комітом запускайте:
     ```bash
     pre-commit run --all-files
     ```
     Це допомагає уникнути помилок (наприклад, `end-of-file-fixer`).
   - Виконуйте коміт:
     ```bash
     git commit -m "Опис змін"
     ```
4. Пуште зміни:
   ```bash
   git push origin feature/назва-задачі
   ```
5. Створюйте Pull Request (PR) на GitHub:
   - Назва: “Опис задачі” (наприклад, “Add EDA notebook”).
   - Опис: Короткий опис змін і призначення ревьювера.
6. Синхронізуйте локальну гілку `main`:
   ```bash
   git checkout main
   git pull origin main
   ```

## Використані бібліотеки

1. `pandas` – Обробка та аналіз даних.
2. `numpy` – Числові обчислення та операції з масивами.
3. `scikit-learn` – Машинне навчання (RandomForest).
4. `xgboost` – Градієнтний бустинг для прогнозування.
5. `lightgbm` – Оптимізований градієнтний бустинг.
6. `joblib` – Збереження та завантаження моделей.
7. `streamlit` – Інтерактивний веб-інтерфейс.
8. `matplotlib` – Візуалізація даних.
9. `seaborn` – Покращена візуалізація даних.
10. `plotly` – Інтерактивні графіки.

## Інструменти розробки

1. `jupyter` – Інтерактивні ноутбуки для аналізу даних.
2. `pre-commit` – Автоматичні перевірки коду перед комітом.
3. `black` – Форматування коду.
4. `flake8` – Перевірка стилю коду.
5. `nbqa` – Інтеграція інструментів перевірки для Jupyter ноутбуків.

## Приклад використання

1. Завантажте CSV-файл із даними клієнтів (шаблон доступний у Streamlit або завантажте з [dataset.csv](посилання_на_Google_Drive)).
2. Отримайте прогноз із ймовірністю відтоку та рекомендаціями.
3. Перегляньте візуалізації (гістограма або індикатор).

Приклад результату:
```
Клієнт (ID: 1001): Висока ймовірність відтоку — 0.75
Рекомендація: Зв’яжіться з клієнтом для пропозиції знижок.
```

## Результати та метрики

1. Модель `RandomForestClassifier` навчена з крос-валідацією (5 фолдів) і параметрами: `n_estimators=368`, `max_depth=3`, `min_samples_split=14`, `min_samples_leaf=9`, `max_features='sqrt'`, `bootstrap=False`.
2. Метрики на тестовій вибірці:
   - Accuracy: 0.9300
   - Precision: 0.9500
   - Recall: 0.9300
   - F1-score: 0.9400
3. Порівняння моделей: `RandomForest` (Accuracy CV: 0.9106) перевершив `XGBoost` (0.8583) і `CatBoost` (0.9358 на тесті, але ризик перенавчання).
4. Аналіз EDA: Висока кореляція `reamining_contract` із відтоком; `bill_avg` видалено через кореляцію -0.45.

## Документація

1. Попередній аналіз (EDA): Див. `notebooks/eda.ipynb` для розподілів ознак (`subscription_age`, `download_avg`, `upload_avg`), обробки пропусків (`reamining_contract`: 21,572), аналізу кореляцій (теплова карта) і викидів (IQR).
2. Передобробка: Пропуски заповнено (`reamining_contract` → 0, `download_avg` → 27.8, `upload_avg` → 2.1), аномалії виправлено (`subscription_age` < 0 → медіана), використано One-Hot Encoding для `download_over_limit`, `StandardScaler` для числових ознак, видалено `id` і `bill_avg` (кореляція -0.45) у `src/preprocessing.py`.
3. Модель: `RandomForestClassifier` (`n_estimators=368`, `max_depth=3`, `min_samples_split=14`, `min_samples_leaf=9`, `max_features='sqrt'`, `bootstrap=False`) обрано після порівняння з `XGBoost` (Accuracy CV: 0.8583) і `CatBoost` (Accuracy тест: 0.9358) за допомогою Optuna (200 trials) у `src/model.py`.
4. Інтерфейс: Streamlit-додаток для введення даних (CSV або вручну) і відображення прогнозів із гістограмами та індикаторами у `src/app.py`.

## Над проєктом працювали:

- Renata Velykholova
- Oleksandr-Kyrychuk
- Arseniy Ishchenko
- Anastasiia Smirnova

