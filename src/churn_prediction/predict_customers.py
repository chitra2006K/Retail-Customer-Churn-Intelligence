
# Phase 8.3: Prepare Customer Data for Prediction
import pandas as pd

customer_df = pd.read_csv("data/processed/customer_features.csv")

print("Customer Data Loaded Successfully!")

X = customer_df[['Frequency', 'Monetary', 'AOV']].copy()

print("\nPrepared Customer Data:")
print(X.head())

print("\nDataset Shape:")
print(X.shape)

print("\nMissing Values:")
print(X.isnull().sum())


# Phase 8.4: Generate Churn Predictions
import joblib

lr_model = joblib.load("models/logistic_regression.joblib")
rf_model = joblib.load("models/random_forest.joblib")

lr_predictions = lr_model.predict(X)
rf_predictions = rf_model.predict(X)

print("\nPredictions Generated Successfully!")

print("\nLogistic Regression Predictions:")
print(lr_predictions[:10])

print("\nRandom Forest Predictions:")
print(rf_predictions[:10])

print("\nTotal Customers Predicted:", len(lr_predictions))


# Phase 8.5: Display Customer-Level Results
results_df = customer_df[['CustomerID', 'Recency']].copy()

results_df['LR_Prediction'] = lr_predictions
results_df['RF_Prediction'] = rf_predictions

results_df['LR_Status'] = results_df['LR_Prediction'].map({
    0: 'Active',
    1: 'Churned'
})

results_df['RF_Status'] = results_df['RF_Prediction'].map({
    0: 'Active',
    1: 'Churned'
})

print("\nCustomer-Level Prediction Results:")
print(results_df.head(10))



# Phase 8.6: Save Prediction Results
results_df.to_csv(
    "data/processed/customer_churn_predictions.csv",
    index=False
)

print("\nPrediction Results Saved Successfully!")

print("File: data/processed/customer_churn_predictions.csv")



# Phase 8.7: Test Complete Prediction System
saved_results = pd.read_csv(
    "data/processed/customer_churn_predictions.csv"
)

print("\n--- Final System Validation ---")

print("Total Records:", len(saved_results))

print("\nMissing Predictions:")
print(saved_results[['LR_Prediction', 'RF_Prediction']].isnull().sum())

print("\nLogistic Regression Valid Labels:",
      saved_results['LR_Prediction'].isin([0, 1]).all())

print("Random Forest Valid Labels:",
      saved_results['RF_Prediction'].isin([0, 1]).all())

if len(saved_results) == len(customer_df):
    print("\nPrediction System Tested Successfully!")
else:
    print("\nRecord Count Mismatch!")

    
