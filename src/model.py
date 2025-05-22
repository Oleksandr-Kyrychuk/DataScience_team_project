import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from preprocessing import preprocess_data
import pickle
import json
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from sklearn.metrics import roc_curve, confusion_matrix


def model_rf() -> RandomForestClassifier:
    """Train RandomForest model and save artifacts."""
    # Project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, "datasets", "internet_service_churn.csv")

    cleaned_data, scaler = preprocess_data(data_path, return_scaler=True)
    X = cleaned_data.drop(columns=["churn"])
    y = cleaned_data["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=368,
        max_depth=3,
        min_samples_split=14,
        min_samples_leaf=9,
        max_features="sqrt",
        bootstrap=False,
    )

    model.fit(X_train, y_train)

    # Cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring="f1")
    print(f"Cross-validation F1 scores: {scores}")
    print(f"Mean F1: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

    # Save model and scaler
    model_path = os.path.join(project_root, "artifacts/model.pkl")
    scaler_path = os.path.join(project_root, "artifacts/scaler.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Evaluate model
    y_pred = model.predict(X_test)  # Обчислюємо y_pred перед використанням
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)  # Використовуємо y_pred після його визначення
    metrics = {}  # Ініціалізуємо словник metrics
    metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    metrics["confusion_matrix"] = cm.tolist()
    metrics_path = os.path.join(project_root, "artifacts/metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Evaluate and save classification report
    metrics.update(classification_report(y_test, y_pred, output_dict=True))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Feature importance
    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values(by="importance", ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=feature_importance)
    plt.title("Feature Importance in Random Forest")
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, "artifacts/feature_importance.png"))
    plt.close()

    return model


if __name__ == "__main__":
    model_rf()
