import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from preprocessing import preprocess_data
import pickle
from sklearn.metrics import classification_report

def model_rf():

    pd.set_option("future.no_silent_downcasting", True)
    data = r'D:\project\internet_service_churn.csv'

    cleaned_data = preprocess_data(data_path=data)

    X = cleaned_data.drop(columns = ['churn'])
    y = cleaned_data['churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    model = RandomForestClassifier(n_estimators=368, max_depth = 3, min_samples_split=14, min_samples_leaf=9, max_features = 'sqrt', bootstrap = False)

    model.fit(X_train, y_train)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    return model

model_rf()
