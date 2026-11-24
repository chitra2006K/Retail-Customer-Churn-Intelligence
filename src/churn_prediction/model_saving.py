
# Phase 7.12: Model Saving
import os
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

customer_df = pd.read_csv("data/processed/customer_features.csv")

customer_df['Churn'] = (customer_df['Recency'] > 90).astype(int)

X = customer_df[['Frequency', 'Monetary', 'AOV']]
y = customer_df['Churn']

lr_model = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic_regression', LogisticRegression(max_iter=1000))
])

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

lr_model.fit(X, y)
rf_model.fit(X, y)

os.makedirs("models", exist_ok=True)

joblib.dump(lr_model, "models/logistic_regression.joblib")
joblib.dump(rf_model, "models/random_forest.joblib")

print("Both models saved successfully!")

print("\nSaved Model Files:")
print("1. models/logistic_regression.joblib")
print("2. models/random_forest.joblib")
