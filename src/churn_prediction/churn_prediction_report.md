# Customer Churn Prediction — Project Report

## 1. Project Overview

The objective of this phase is to develop Machine Learning models that predict whether a customer is active or churned based on their purchasing behaviour.

Two classification algorithms were implemented:
- Logistic Regression
- Random Forest Classifier

## 2. Churn Definition

For this project, customers who have not made a purchase for more than 90 days are classified as churned.

- 0 = Active Customer (Recency ≤ 90 days)
- 1 = Churned Customer (Recency > 90 days)

This is a project-defined proxy for churn, not independently verified customer departure.

## 3. Churn Distribution

Total customers: 4,338

| Customer Type | Count |
|---|---:|
| Active Customers | 2,889 |
| Churned Customers | 1,449 |

## 4. Feature Selection

Three input features were selected:

- Frequency: Number of purchases made by a customer.
- Monetary: Total amount spent by a customer.
- AOV: Average Order Value.

Target variable:
- Churn: 0 or 1.

Recency was excluded from the input features because it was used to create the churn labels. Including it would cause target leakage.

## 5. Train-Test Split

The dataset was divided into:

| Dataset | Records |
|---|---:|
| Training Data (80%) | 3,470 |
| Testing Data (20%) | 868 |
| Total | 4,338 |

Stratified sampling was used to preserve the class distribution.

## 6. Machine Learning Models

### Logistic Regression
A classification algorithm used to predict whether a customer belongs to the active or churned category. StandardScaler was applied through a Pipeline.

### Random Forest
An ensemble learning algorithm that combines predictions from multiple decision trees. The model was configured with 100 trees.

## 7. Model Performance Comparison

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 73.16% | 64.75% |
| Precision (Churn) | 60.52% | 46.95% |
| Recall (Churn) | 56.55% | 42.41% |
| F1-Score (Churn) | 58.47% | 44.57% |

## 8. Confusion Matrix Results

### Logistic Regression

| Actual / Predicted | Active | Churned |
|---|---:|---:|
| Active | 471 | 107 |
| Churned | 126 | 164 |

### Random Forest

| Actual / Predicted | Active | Churned |
|---|---:|---:|
| Active | 439 | 139 |
| Churned | 167 | 123 |

Logistic Regression correctly classified 635 test customers, while Random Forest correctly classified 562.

## 9. Feature Importance

Random Forest feature importance results:

| Feature | Importance |
|---|---:|
| Monetary | 0.478254 |
| AOV | 0.416985 |
| Frequency | 0.104761 |

Monetary had the highest importance score, followed by AOV and Frequency.

These scores represent the model's relative feature importance and do not establish causation.

## 10. Final Model Selection

Logistic Regression performed better than Random Forest across all four evaluated metrics.

Therefore, Logistic Regression was selected as the better-performing model among the two tested algorithms.

## 11. Model Saving

Both trained models were saved using the Joblib library:

- `models/logistic_regression.joblib`
- `models/random_forest.joblib`

The final saved models were trained using all 4,338 customer records after the holdout evaluation was completed.

## 12. Conclusion

The customer churn prediction phase successfully implemented two classification algorithms and evaluated their performance.

Logistic Regression achieved 73.16% accuracy and a churn-class F1-score of 58.47%, outperforming Random Forest.

The results demonstrate that purchasing behaviour features can help predict the project's rule-based churn labels. However, since churn was defined using a 90-day inactivity threshold, further validation against independently observed customer churn would be needed before real-world deployment.


