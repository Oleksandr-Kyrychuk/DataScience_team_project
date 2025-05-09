import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from catboost import CatBoostClassifier, Pool, cv
import matplotlib.pyplot as plt
from preprocessing import preprocess_data

# Налаштування pandas
pd.set_option("future.no_silent_downcasting", True)

# Шлях до даних
data_path = "D:/DataScience_team_project/datasets/internet_service_churn.csv"

# Препроцесинг даних за допомогою функції з preprocessing.py
df_churn = preprocess_data(data_path=data_path)

# Розділення даних
X = df_churn.drop(columns=["churn"])
y = df_churn["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ініціалізація моделі
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    auto_class_weights="Balanced",
    verbose=100,
    eval_metric="AUC",
)

# Навчання з оцінкою на тестовій вибірці
model.fit(X_train, y_train, eval_set=(X_test, y_test))

# Прогнозування
y_train_pred_proba = model.predict_proba(X_train)[:, 1]
y_test_pred_proba = model.predict_proba(X_test)[:, 1]
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Обчислення AUC-ROC
auc_train = roc_auc_score(y_train, y_train_pred_proba)
auc_test = roc_auc_score(y_test, y_test_pred_proba)
print(f"AUC-ROC на тренувальній вибірці: {auc_train:.4f}")
print(f"AUC-ROC на тестовій вибірці: {auc_test:.4f}")

# Обчислення F1-score і Confusion Matrix
print(f"F1-score на тренувальній вибірці: {f1_score(y_train, y_train_pred):.4f}")
print(f"F1-score на тестовій вибірці: {f1_score(y_test, y_test_pred):.4f}")
print("Confusion Matrix (тренувальна вибірка):")
print(confusion_matrix(y_train, y_train_pred))
print("Confusion Matrix (тестова вибірка):")
print(confusion_matrix(y_test, y_test_pred))

# Важливість ознак
feature_importance = pd.DataFrame(
    {"feature": X.columns, "importance": model.get_feature_importance()}
).sort_values(by="importance", ascending=False)
print("\nВажливість ознак:")
print(feature_importance)

# Крос-валідація
pool = Pool(X, y)
params = {
    "iterations": 500,
    "learning_rate": 0.1,
    "depth": 6,
    "auto_class_weights": "Balanced",
    "loss_function": "Logloss",
    "eval_metric": "AUC",
}
cv_results = cv(pool, params, fold_count=5, verbose=100)
print(f"\nСереднє AUC-ROC на крос-валідації: {cv_results['test-AUC-mean'].iloc[-1]:.4f}")
print(f"Стандартне відхилення AUC-ROC: {cv_results['test-AUC-std'].iloc[-1]:.4f}")

# Криві навчання — побудова вручну
fractions = np.linspace(0.1, 0.99, 10)  # Змінили 1.0 на 0.99
train_scores, test_scores = [], []

for frac in fractions:
    X_frac, _, y_frac, _ = train_test_split(
        X_train, y_train, train_size=frac, stratify=y_train, random_state=42
    )

    model_frac = CatBoostClassifier(
        iterations=500, learning_rate=0.1, depth=6, auto_class_weights="Balanced", verbose=0
    )
    model_frac.fit(X_frac, y_frac)

    y_frac_pred = model_frac.predict_proba(X_frac)[:, 1]
    y_test_pred = model_frac.predict_proba(X_test)[:, 1]

    train_scores.append(roc_auc_score(y_frac, y_frac_pred))
    test_scores.append(roc_auc_score(y_test, y_test_pred))

# Побудова графіка
train_sizes = [int(frac * len(X_train)) for frac in fractions]

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores, label="Train AUC")
plt.plot(train_sizes, test_scores, label="Test AUC")
plt.xlabel("Training Examples")
plt.ylabel("AUC-ROC")
plt.title("Learning Curves (CatBoost, Manual)")
plt.legend()
plt.grid()
plt.show()
