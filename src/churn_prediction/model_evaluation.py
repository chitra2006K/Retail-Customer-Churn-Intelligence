
# Phase 7.9: Confusion Matrix
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

lr_cm = confusion_matrix(y_test, lr_pred, labels=[0, 1])
rf_cm = confusion_matrix(y_test, rf_pred, labels=[0, 1])

print("Logistic Regression Confusion Matrix:")
print(lr_cm)

print("\nRandom Forest Confusion Matrix:")
print(rf_cm)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay(
    confusion_matrix=lr_cm,
    display_labels=['Active', 'Churned']
).plot(ax=axes[0], cmap='Blues', colorbar=False)

axes[0].set_title("Logistic Regression")

ConfusionMatrixDisplay(
    confusion_matrix=rf_cm,
    display_labels=['Active', 'Churned']
).plot(ax=axes[1], cmap='Greens', colorbar=False)

axes[1].set_title("Random Forest")

plt.tight_layout()
plt.show()


