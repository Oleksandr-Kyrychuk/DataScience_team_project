import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from preprocessing import preprocess_data
import pickle
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt
import seaborn as sns


def model_rf():
    # Визначаємо корінь проєкту (на один рівень вище від src)
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Поточна директорія (src)
    project_root = os.path.dirname(current_dir)  # Корінь проєкту (на один рівень вище)

    data_path = os.path.join(project_root, "datasets", "internet_service_churn.csv")

    cleaned_data, scaler = preprocess_data(data_path, return_scaler=True)

    X = cleaned_data.drop(columns=["churn"])
    y = cleaned_data["churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
        n_estimators=368,
        max_depth=3,
        min_samples_split=14,
        min_samples_leaf=9,
        max_features="sqrt",
        bootstrap=False,
    )

    model.fit(X_train, y_train)

    # Зберігаємо модель і scaler у корені проєкту
    model_path = os.path.join(project_root, "model.pkl")
    scaler_path = os.path.join(project_root, "scaler.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Аналіз важливості ознак
    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values(by="importance", ascending=False)

    print("\nFeature Importance:")
    print(feature_importance)

    # Візуалізація важливості ознак (зберігаємо в корені проєкту)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=feature_importance)
    plt.title("Feature Importance in Random Forest")
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, "feature_importance.png"))
    plt.close()

    return model


if __name__ == "__main__":
    model_rf()
