import os
import re
import time
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# =========================================================
#  Data Loading & Preprocessing
# =========================================================
def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    print(f" Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    numeric_cols = ['heart_rate', 'bp_systolic', 'bp_diastolic',
                    'temperature', 'respiratory_rate', 'oxygen_saturation']

    # Handle missing values
    imputer = KNNImputer(n_neighbors=5)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # Derived features
    df['bp_risk_score'] = (df['bp_systolic'] - 120) / 20 + (df['bp_diastolic'] - 80) / 10
    df['vital_composite'] = (
        (df['heart_rate'] - 70) / 30 +
        (df['temperature'] - 98.6) / 2 +
        (100 - df['oxygen_saturation']) / 10
    )

    # Encode categorical
    le_priority = LabelEncoder()
    df['triage_priority_encoded'] = le_priority.fit_transform(df['triage_priority'])

    le_gender = LabelEncoder()
    df['gender_encoded'] = le_gender.fit_transform(df['gender'])

    # Feature scaling
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print(" Data preprocessing completed")
    return df, numeric_cols, scaler


# =========================================================
#  NLP Processing
# =========================================================
def preprocess_symptoms(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def extract_symptom_features(df: pd.DataFrame, max_features=100):
    df['symptoms_cleaned'] = df['chief_complaint'].apply(preprocess_symptoms)
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    symptom_features = tfidf.fit_transform(df['symptoms_cleaned'])

    symptom_df = pd.DataFrame(
        symptom_features.toarray(),
        columns=[f'symptom_tfidf_{i}' for i in range(symptom_features.shape[1])]
    )
    df = pd.concat([df.reset_index(drop=True), symptom_df], axis=1)
    print(f" Extracted {symptom_features.shape[1]} symptom features")
    return df, tfidf


# =========================================================
#  Model Training
# =========================================================
def train_models(df, numeric_cols):
    feature_cols = numeric_cols + [f'symptom_tfidf_{i}' for i in range(100)] + \
                   ['vital_composite', 'bp_risk_score', 'gender_encoded', 'age']
    X = df[feature_cols]
    y = df['triage_priority_encoded']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f" Training samples: {X_train.shape[0]}")
    print(f" Validation samples: {X_val.shape[0]}")
    print(f" Testing samples: {X_test.shape[0]}")

    # Random Forest Grid Search
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced']
    }

    rf_model = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf_model, rf_params, cv=3, scoring='f1_weighted', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    print(f" Best Random Forest params: {rf_grid.best_params_}")

    # XGBoost
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    # Gradient Boosting
    gb_model = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=7, random_state=42
    )
    gb_model.fit(X_train, y_train)

    return {
        'Random Forest': best_rf,
        'XGBoost': xgb_model,
        'Gradient Boosting': gb_model
    }, (X_train, X_val, X_test, y_train, y_val, y_test)


# =========================================================
#  Evaluation & Visualization
# =========================================================
def evaluate_models(models, X_val, y_val):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average='weighted'
        )
        results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}
        print(f"\n {name} Validation Performance:")
        print(f"Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
    return results


def visualize_feature_importance(model, feature_cols):
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title("Top 15 Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300)
    plt.show()


# =========================================================
#  Save Models
# =========================================================
def save_artifacts(model, scaler, tfidf):
    with open("telemedicine_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    print("All artifacts saved successfully.")


# =========================================================
# 🏁 Main Execution
# =========================================================
if __name__ == "__main__":
    start_time = time.time()
    df, numeric_cols, scaler = load_and_preprocess_data("rural_healthcare_data.csv")
    df, tfidf = extract_symptom_features(df)
    models, data_splits = train_models(df, numeric_cols)
    X_train, X_val, X_test, y_train, y_val, y_test = data_splits

    results = evaluate_models(models, X_val, y_val)
    best_model_name = max(results, key=lambda x: results[x]['F1'])
    best_model = models[best_model_name]
    print(f"\n Best Model Selected: {best_model_name}")

    visualize_feature_importance(best_model, numeric_cols + [f"symptom_tfidf_{i}" for i in range(100)])
    save_artifacts(best_model, scaler, tfidf)
    print(f"Pipeline completed in {(time.time() - start_time):.2f} seconds.")
