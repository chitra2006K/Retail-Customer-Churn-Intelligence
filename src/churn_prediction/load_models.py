
# Phase 8.1: Load Saved ML Models

import joblib

# Load Logistic Regression model
lr_model = joblib.load("models/logistic_regression.joblib")

# Load Random Forest model
rf_model = joblib.load("models/random_forest.joblib")

print("Models Loaded Successfully!")

print("\nLoaded Models:")
print("1. Logistic Regression")
print("2. Random Forest")


# Phase 8.2: Verify Loaded Models
import pandas as pd

customer_df = pd.read_csv("data/processed/customer_features.csv")

X_sample = customer_df[['Frequency', 'Monetary', 'AOV']].head(5)

lr_predictions = lr_model.predict(X_sample)

rf_predictions = rf_model.predict(X_sample)

print("\nLogistic Regression Predictions:")
print(lr_predictions)

print("\nRandom Forest Predictions:")
print(rf_predictions)


