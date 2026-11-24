
# Phase 7.1: Define Customer Churn

import pandas as pd

# Define churn threshold
CHURN_THRESHOLD = 90

print("Customer Churn Definition")
print("-------------------------")
print(f"Customers inactive for more than {CHURN_THRESHOLD} days will be considered churned.")

print("\nChurn Labels:")
print("0 = Active Customer")
print("1 = Churned Customer")


# Phase 7.2: Create Churn Labels

customer_df = pd.read_csv("data/processed/customer_features.csv")

customer_df['Churn'] = (customer_df['Recency'] > CHURN_THRESHOLD).astype(int)

print("\nChurn Labels Created Successfully!")
print(customer_df[['CustomerID', 'Recency', 'Churn']].head())

print("\nChurn Distribution:")
print(customer_df['Churn'].value_counts())


# Phase 7.3: Churn Analysis
print("\nChurn Analysis:")
print(customer_df.groupby('Churn')[['Recency', 'Frequency', 'Monetary', 'AOV']].mean())



# Phase 7.4: Feature Selection
X = customer_df[['Frequency', 'Monetary', 'AOV']]

y = customer_df['Churn']

print("\nFeature Selection Completed!")

print("\nInput Features (X):")
print(X.head())

print("\nTarget Variable (y):")
print(y.head())

print("\nInput Shape:", X.shape)
print("Target Shape:", y.shape)



# Phase 7.5: Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split Completed!")

print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)

print("Training Labels:", y_train.shape)
print("Testing Labels:", y_test.shape)


