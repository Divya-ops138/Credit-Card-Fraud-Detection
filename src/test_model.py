import os
import joblib
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "test_transactions.csv")

# Load model
model = xgb.XGBClassifier()
model.load_model(
    os.path.join(MODEL_DIR, "xgboost_fraud_model.json")
)

# Load preprocessing files
preprocessor = joblib.load(
    os.path.join(MODEL_DIR, "preprocessor.pkl")
)

input_columns = joblib.load(
    os.path.join(MODEL_DIR, "input_columns.pkl")
)

# Load test data
df = pd.read_csv(DATA_PATH)

X = df[input_columns]
y = df["isFraud"]

# Preprocess
X_processed = preprocessor.transform(X)

# Predictions

# First calculate probabilities
probabilities = model.predict_proba(X_processed)[:, 1]

# Apply optimized threshold
OPTIMAL_THRESHOLD = 0.8673788

predictions = (
    probabilities >= OPTIMAL_THRESHOLD
).astype(int)

# Evaluation
print("\nClassification Report:")
print(classification_report(y, predictions))

print("Confusion Matrix:")
print(confusion_matrix(y, predictions))

print("ROC-AUC:", roc_auc_score(y, probabilities))
print("PR-AUC:", average_precision_score(y, probabilities))