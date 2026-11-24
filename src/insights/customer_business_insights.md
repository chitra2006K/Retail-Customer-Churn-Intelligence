
# Customer Segmentation and Churn Prediction
## Business Insights Report

### 1. Project Overview
This project uses Machine Learning techniques to segment
customers and predict customer churn using Online Retail data.

A total of 4,338 customers were analyzed.

### 2. Customer Segmentation Results

Four customer segments were identified using K-Means clustering:

- Regular Customers: 3,228
- Low-Engagement Customers: 1,083
- Highly Active High-Value Customers: 25
- Exceptional High-Value Customers: 2

### 3. Churn Prediction Results

Two machine learning models were developed:

- Logistic Regression
- Random Forest

Logistic Regression achieved approximately 73.16% accuracy.
Random Forest achieved approximately 64.75% accuracy.

Logistic Regression performed better on the evaluated test metrics.

### 4. Churn Analysis

Predicted churn rates across customer segments:

| Customer Segment | Logistic Regression | Random Forest |
|---|---:|---:|
| Exceptional High-Value | 50.00% | 50.00% |
| Highly Active High-Value | 0.00% | 0.00% |
| Low-Engagement | 66.85% | 99.17% |
| Regular | 23.79% | 11.56% |

Low-Engagement Customers showed the highest predicted churn rate.

### 5. High-Value Customer Risk

- Total high-value customers: 27
- Customers predicted to churn by both models: 1
- At-risk customer ID: 12346
- Customer spending: 77,183.60

This customer may require personalised retention attention.

### 6. Customer Retention Recommendations

- Low-Engagement Customers: Win-back offers and discounts.
- Regular Customers: Loyalty rewards and personalised offers.
- Exceptional High-Value Customers: VIP support and retention offers.
- Highly Active High-Value Customers: Premium services and exclusive benefits.

### 7. Final Business Insights

- Regular Customers form the largest customer segment.
- Low-Engagement Customers have the highest predicted churn rate.
- One high-value customer was flagged by both models.
- Logistic Regression performed better than Random Forest on the evaluated metrics.
- Customer segmentation and churn prediction can help businesses plan targeted retention strategies.

### 8. Conclusion

The project combines customer segmentation and churn prediction
to provide useful business insights.

It helps identify different customer groups, estimate churn risk,
and suggest suitable retention strategies.

The predictions are estimates based on the available dataset
and should not be treated as confirmed customer behaviour.
