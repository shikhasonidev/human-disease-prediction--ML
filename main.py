import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    data = data.dropna().drop_duplicates()
    X = data.drop("outcome", axis=1)
    y = data["outcome"]

    X['sex'] = LabelEncoder().fit_transform(X['sex'])

    numeric_cols = ['age', 'glucose', 'bmi', 'blood_pressure', 'cholesterol']
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y, scaler

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    data = load_data("disease_data.csv")
    X, y, scaler = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    joblib.dump(model, "disease_model.pkl")
