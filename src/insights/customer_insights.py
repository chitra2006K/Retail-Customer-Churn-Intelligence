
# Phase 9.1: Load Customer Data
import pandas as pd

segments_df = pd.read_csv(
    "data/processed/customer_segments.csv"
)

predictions_df = pd.read_csv(
    "data/processed/customer_churn_predictions.csv"
)

print("Customer Segmentation Data Loaded!")
print("Churn Prediction Data Loaded!")

print("\nSegmentation Data:")
print(segments_df.head())

print("\nChurn Prediction Data:")
print(predictions_df.head())

print("\nSegmentation Shape:", segments_df.shape)
print("Prediction Shape:", predictions_df.shape)


# Phase 9.2: Combine Customer Data
customer_df = pd.merge(
    segments_df,
    predictions_df[['CustomerID', 'LR_Prediction', 'RF_Prediction',
                    'LR_Status', 'RF_Status']],
    on='CustomerID',
    how='inner'
)

print("\nCustomer Data Combined Successfully!")

print("\nCombined Customer Data:")
print(customer_df.head())

print("\nCombined Dataset Shape:")
print(customer_df.shape)

print("\nTotal Customers:", customer_df['CustomerID'].nunique())



# Phase 9.3: Analyze Churn Across Customer Segments
segment_analysis = customer_df.groupby('Segment').agg(
    Total_Customers=('CustomerID', 'count'),
    LR_Predicted_Churn=('LR_Prediction', 'sum'),
    RF_Predicted_Churn=('RF_Prediction', 'sum')
).reset_index()

segment_analysis['LR_Churn_Rate'] = (
    segment_analysis['LR_Predicted_Churn'] /
    segment_analysis['Total_Customers'] * 100
)

segment_analysis['RF_Churn_Rate'] = (
    segment_analysis['RF_Predicted_Churn'] /
    segment_analysis['Total_Customers'] * 100
)

print("\nChurn Analysis Across Customer Segments:")
print(segment_analysis.round(2))



# Phase 9.4: Identify High-Value Customers at Churn Risk
high_value_customers = customer_df[
    customer_df['Segment'].isin([
        'Exceptional High-Value Customers',
        'Highly Active High-Value Customers'
    ])
].copy()

high_value_at_risk = high_value_customers[
    (high_value_customers['LR_Prediction'] == 1) &
    (high_value_customers['RF_Prediction'] == 1)
]

print("\nHigh-Value Customer Analysis:")
print(high_value_customers[
    ['CustomerID', 'Segment', 'Monetary', 'LR_Status', 'RF_Status']
])

print("\nHigh-Value Customers Predicted to Churn by Both Models:")
print(high_value_at_risk[
    ['CustomerID', 'Segment', 'Monetary', 'LR_Status', 'RF_Status']
])

print("\nTotal High-Value Customers:", len(high_value_customers))
print("High-Value Customers at Risk:", len(high_value_at_risk))


# Phase 9.5: Customer Retention Recommendations
print("\nCustomer Retention Recommendations:")

recommendations = {
    'Low-Engagement Customers':
        'Send win-back offers and personalised discounts.',

    'Regular Customers':
        'Provide loyalty rewards to encourage repeat purchases.',

    'Exceptional High-Value Customers':
        'Provide VIP support and personalised retention offers.',

    'Highly Active High-Value Customers':
        'Maintain premium services and exclusive benefits.'
}

for segment, recommendation in recommendations.items():
    print(f"\n{segment}:")
    print(recommendation)



# Phase 9.6: Generate Business Insights
print("\n========== BUSINESS INSIGHTS ==========")

highest_churn_segment = segment_analysis.loc[
    segment_analysis['LR_Churn_Rate'].idxmax()
]

largest_segment = segment_analysis.loc[
    segment_analysis['Total_Customers'].idxmax()
]

lr_total_churn = customer_df['LR_Prediction'].sum()
rf_total_churn = customer_df['RF_Prediction'].sum()

print("\n1. Highest Predicted Churn Segment:")
print(highest_churn_segment['Segment'])
print("Churn Rate:", round(highest_churn_segment['LR_Churn_Rate'], 2), "%")

print("\n2. Largest Customer Segment:")
print(largest_segment['Segment'])
print("Total Customers:", largest_segment['Total_Customers'])

print("\n3. High-Value Customers at Risk:")
print(len(high_value_at_risk))

print("\n4. Total Churn Predictions:")
print("Logistic Regression:", lr_total_churn)
print("Random Forest:", rf_total_churn)


