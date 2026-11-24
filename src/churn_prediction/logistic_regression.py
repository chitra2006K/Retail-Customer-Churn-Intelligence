
# Phase 7.6: Logistic Regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

customer_df = pd.read_csv("data/processed/customer_features.csv")

customer_df['Churn'] = (customer_df['Recency'] > 90).astype(int)

X = customer_df[['Frequency', 'Monetary', 'AOV']]
y = customer_df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic_regression', LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

print("Logistic Regression Model Trained Successfully!")




# Phase 7.7: Logistic Regression Predictions
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nLogistic Regression Predictions:")
print(y_pred[:10])

print("\nActual Churn Labels:")
print(y_test.iloc[:10].values)

print("\nModel Accuracy:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

