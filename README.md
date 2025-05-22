# DataScience_team_project: Customer Churn Prediction

This project is designed to predict the likelihood of customer churn for a telecommunications company based on their data. Using a RandomForest model, the project identifies customers at high risk of leaving and provides recommendations for retention. The Streamlit interface allows uploading CSV files or manually entering data, with results displayed alongside visualizations (histograms, indicators). The project is packaged in Docker for easy deployment.

## Data Description

1. Dataset `internet_service_churn.csv` – CSV file with customer data from a telecommunications company.
2. Features:
   - `id`: Unique user identifier.
   - `is_tv_subscriber`: TV service subscription (0 or 1).
   - `is_movie_package_subscriber`: Movie package subscription (0 or 1).
   - `subscription_age`: Subscription duration in months.
   - `bill_avg`: Average monthly bill.
   - `remaining_contract`: Remaining contract duration in months (21,572 missing values).
   - `service_failure_count`: Number of complaints/connection issues.
   - `download_avg`: Average download speed (381 missing values).
   - `upload_avg`: Average upload speed (381 missing values).
   - `download_over_limit`: Data limit exceeded (0 or 1).
   - `churn`: Customer churn (0 – no, 1 – yes, 55.41% positive class).
3. Source: A sample dataset is provided in `datasets/internet_service_churn.csv` for testing purposes.

## Key Features

1. Predicting customer churn probability using RandomForest.
2. Interactive Streamlit interface for uploading CSV or manual data entry.
3. Result visualization: histograms for multiple customers, indicators for single customers.
4. Containerization via Docker and Docker Compose.
5. Support for English and Ukrainian languages.

## Project Structure

1. `datasets/internet_service_churn.csv` – Sample dataset for prediction.
2. `docs/` – Documentation and images (screenshots, diagrams).
3. `notebooks/` – Jupyter notebooks for analysis:
   - `churn.ipynb` – Exploratory data analysis.
4. `src/` – Main code:
   - `preprocessing.py` – Data preprocessing.
   - `app.py` – Streamlit interface.
   - `inference.py` – Prediction logic.
   - `model.py` – Model training.
   - `ui_components.py` – UI components for Streamlit.
   - `prediction.py` – Prediction functions.
   - `session_manager.py` – Session state and configuration management.
5. `Dockerfile` – Docker image configuration.
6. `docker-compose.yml` – Docker Compose setup.
7. `requirements.txt` – List of dependencies.
8. `README.md` – Project description.

## Requirements

1. Python 3.12 (recommended for compatibility, minimum 3.10 for Docker).
2. Docker and Docker Compose for containerization.
3. Dependencies installed from `requirements.txt`.
4. Sample dataset in `datasets/internet_service_churn.csv`.

## Getting Started

1. Clone the repository:
   ```bash
   git clone git@github.com:Oleksandr-Kyrychuk/DataScience_team_project.git
   ```
2. Create a branch:
   ```bash
   git checkout -b feature/your-branch-name
   ```
3. Ensure the sample dataset is available in `datasets/internet_service_churn.csv`.

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the Streamlit app:
   ```bash
   streamlit run src/app.py
   ```

## Environment Setup

1. Ensure Python 3.12 (or 3.10 for Docker compatibility) is installed. If not, download from [python.org](https://www.python.org/downloads/) and install.
2. Create a virtual environment:
   ```bash
   py -3.12 -m venv .venv  # Windows
   python3.10 -m venv .venv  # Linux/Mac
   ```
3. Activate the environment:
   ```bash
   .\.venv\Scripts\Activate.ps1  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Set up `pre-commit` for automatic checks:
   ```bash
   pre-commit install
   ```

## Running with Docker

1. Ensure Docker and Docker Compose are installed.
2. Run the project:
   ```bash
   docker-compose up --build
   ```
3. Open `http://localhost:8501` in your browser.
4. Stop the container:
   ```bash
   docker-compose down
   ```

## Containerization

1. `Dockerfile` builds an image based on Python 3.10, copies the project code, installs dependencies from `requirements.txt`, runs `src/model.py` to prepare the model, and then `src/app.py` for Streamlit on port 8501.
2. `docker-compose.yml` automates container startup with a single `docker-compose up --build` command, configuring the network and port 8501.
3. Containerization ensures reproducibility and easy deployment on any system with Docker.

## Workflow

1. Create a new branch for each task:
   ```bash
   git checkout -b feature/task-name
   ```
   Example:
   ```bash
   git checkout -b feature/eda-analysis
   ```
2. Stage changed files:
   ```bash
   git add .  # or specific file
   ```
3. Commit changes:
   - Before committing, run:
     ```bash
     pre-commit run --all-files
     ```
     This helps avoid errors (e.g., `end-of-file-fixer`).
   - Perform the commit:
     ```bash
     git commit -m "Description of changes"
     ```
4. Push changes:
   ```bash
   git push origin feature/task-name
   ```
5. Create a Pull Request (PR) on GitHub:
   - Title: “Task description” (e.g., “Add EDA notebook”).
   - Description: Brief summary of changes and reviewer assignment.
6. Sync the local `main` branch:
   ```bash
   git checkout main
   git pull origin main
   ```

## Libraries Used

1. `pandas` – Data processing and analysis.
2. `numpy` – Numerical computations and array operations.
3. `scikit-learn` – Machine learning (RandomForest).
4. `xgboost` – Gradient boosting for prediction.
5. `lightgbm` – Optimized gradient boosting.
6. `joblib` – Model saving and loading.
7. `streamlit` – Interactive web interface.
8. `matplotlib` – Data visualization.
9. `seaborn` – Enhanced data visualization.
10. `plotly` – Interactive charts.
11. `pyyaml` – Configuration file parsing.

## Development Tools

1. `jupyter` – Interactive notebooks for data analysis.
2. `pre-commit` – Automatic code checks before commits.
3. `black` – Code formatting.
4. `flake8` – Code style checking.
5. `nbqa` – Integration of checking tools for Jupyter notebooks.

## Usage Example

1. Upload the sample CSV file (`datasets/internet_service_churn.csv`) or enter data manually via the Streamlit interface.
2. Receive churn probability predictions and recommendations.
3. View visualizations (histogram for multiple customers or indicator for a single customer).

Example output:
```
Customer (ID: 1001): High churn probability — 0.75
Recommendation: Contact the customer to offer discounts.
```

## Results and Metrics

1. The `RandomForestClassifier` model was trained with 5-fold cross-validation and parameters: `n_estimators=368`, `max_depth=3`, `min_samples_split=14`, `min_samples_leaf=9`, `max_features='sqrt'`, `bootstrap=False`.
2. Metrics on the test set:
   - Accuracy: 0.9300
   - Precision: 0.9500
   - Recall: 0.9300
   - F1-score: 0.9400
3. Model comparison: `RandomForest` (Accuracy CV: 0.9106) outperformed `XGBoost` (0.8583) and `CatBoost` (0.9358 on test, but risk of overfitting).
4. EDA findings: High correlation of `remaining_contract` with churn; `bill_avg` removed due to -0.45 correlation.

## Documentation

1. Exploratory Data Analysis (EDA): See `notebooks/eda.ipynb` for feature distributions (`subscription_age`, `download_avg`, `upload_avg`), handling missing values (`remaining_contract`: 21,572), correlation analysis (heatmap), and outlier detection (IQR).
2. Preprocessing: Missing values filled (`remaining_contract` → 0, `download_avg` → 27.8, `upload_avg` → 2.1), anomalies corrected (`subscription_age` < 0 → median), One-Hot Encoding for `download_over_limit`, `StandardScaler` for numerical features, `id` and `bill_avg` removed (correlation -0.45) in `src/preprocessing.py`.
3. Model: `RandomForestClassifier` (`n_estimators=368`, `max_depth=3`, `min_samples_split=14`, `min_samples_leaf=9`, `max_features='sqrt'`, `bootstrap=False`) selected after comparison with `XGBoost` (Accuracy CV: 0.8583) and `CatBoost` (Accuracy test: 0.9358) using Optuna (200 trials) in `src/model.py`.
4. Interface: Streamlit app for data input (CSV or manual) and displaying predictions with histograms and indicators in `src/app.py`.

## Contributors

- Renata Velykholova
- Oleksandr-Kyrychuk
- Arseniy Ishchenko
- Anastasiia Smirnova



![CI Status](https://github.com/Oleksandr-Kyrychuk/DataScience_team_project/actions/workflows/ci.yml/badge.svg)
