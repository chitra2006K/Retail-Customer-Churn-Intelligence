
# Phase 7.10: Model Comparison
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

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

lr_model = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic_regression', LogisticRegression(max_iter=1000))
])

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

results = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],

    'Logistic Regression': [
        accuracy_score(y_test, lr_pred),
        precision_score(y_test, lr_pred, pos_label=1),
        recall_score(y_test, lr_pred, pos_label=1),
        f1_score(y_test, lr_pred, pos_label=1)
    ],

    'Random Forest': [
        accuracy_score(y_test, rf_pred),
        precision_score(y_test, rf_pred, pos_label=1),
        recall_score(y_test, rf_pred, pos_label=1),
        f1_score(y_test, rf_pred, pos_label=1)
    ]
}

comparison_df = pd.DataFrame(results)

print("\nModel Performance Comparison:")
print(comparison_df.to_string(index=False))


