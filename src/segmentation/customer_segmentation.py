import pandas as pd

# Load customer-level features
file_path = "data/processed/customer_features.csv"

customer_df = pd.read_csv(file_path)

print("Customer features loaded successfully!")
print(customer_df.head())

# Select features for customer segmentation
segmentation_features = [
    'Recency',
    'Frequency',
    'Monetary',
    'AOV'
]

X = customer_df[segmentation_features].copy()

print("\nFeatures selected for segmentation:")
print(X.head())

# Check for missing values
print("\nMissing values in segmentation features:")
print(X.isnull().sum())

# Scale the segmentation features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaled segmentation data:")
print(X_scaled[:5])

# Find the optimal number of clusters using the Elbow Method
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss = []

for k in range(2, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)

print("\nWCSS values:")
for k, value in zip(range(2, 11), wcss):
    print(f"K = {k}: WCSS = {value:.2f}")

plt.figure(figsize=(8, 5))

plt.plot(range(2, 11), wcss, marker='o')

plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal Number of Clusters')

plt.xticks(range(2, 11))
plt.grid(True)

plt.show()    


# Train the final K-Means model
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)

print("\nK-Means model trained successfully!")


# Assign each customer to a cluster
customer_df['Cluster'] = kmeans.labels_

print("\nCluster labels assigned successfully!")
print(customer_df[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'AOV', 'Cluster']].head())


# Analyze the characteristics of each cluster
cluster_summary = customer_df.groupby('Cluster')[
    ['Recency', 'Frequency', 'Monetary', 'AOV']
].mean()

print("\nCluster Summary:")
print(cluster_summary.round(2))


# Assign meaningful business names to the clusters
cluster_names = {
    0: 'Exceptional High-Value Customers',
    1: 'Regular Customers',
    2: 'Low-Engagement Customers',
    3: 'Highly Active High-Value Customers'
}

customer_df['Segment'] = customer_df['Cluster'].map(cluster_names)

print("\nFinal Customer segments:")
print(customer_df[['CustomerID', 'Cluster', 'Segment']].head())


# Display cluster characteristics for business interpretation
print("\nCustomer Segment Analysis:")

for cluster in sorted(customer_df['Cluster'].unique()):
    cluster_data = cluster_summary.loc[cluster]

    print(f"\nCluster {cluster}:")
    print(f"Average Recency: {cluster_data['Recency']:.2f}")
    print(f"Average Frequency: {cluster_data['Frequency']:.2f}")
    print(f"Average Monetary: £{cluster_data['Monetary']:.2f}")
    print(f"Average AOV: £{cluster_data['AOV']:.2f}")


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))

for cluster in sorted(customer_df['Cluster'].unique()):

    cluster_data = customer_df[
        customer_df['Cluster'] == cluster
    ]

    plt.scatter(
        cluster_data['Recency'],
        cluster_data['Monetary'],
        label=f'Cluster {cluster}',
        alpha=0.6
    )

plt.xlabel('Recency (Days)')
plt.ylabel('Monetary (£) - Log Scale')
plt.title('Customer Segmentation: Recency vs Monetary')

plt.yscale('symlog', linthresh=100)

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



# Phase 6.11: Save customer segmentation results
import os
# Create the output directory if it does not exist
os.makedirs("data/processed", exist_ok=True)

# Save customer segmentation results
output_path = "data/processed/customer_segments.csv"

customer_df.to_csv(output_path, index=False)
print("\nCustomer segmentation results saved successfully!")
print(f"File location: {output_path}")



# Phase 6.12: Validate customer segment sizes

# Count customers in each cluster
cluster_counts = customer_df['Cluster'].value_counts().sort_index()

print("\nCustomer Count in Each Cluster:")
print(cluster_counts)

# Calculate percentage of customers in each cluster
cluster_percentage = (
    customer_df['Cluster'].value_counts(normalize=True).sort_index() * 100
)

print("\nCustomer Percentage in Each Cluster:")
print(cluster_percentage.round(2))


# Phase 6.13.1: Investigate small clusters

small_clusters = customer_df[customer_df['Cluster'].isin([0, 3])]

print("\nCustomers in Small Clusters:")
print(small_clusters[
    ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'AOV', 'Cluster']
])


# Phase 6.13.2: Check extreme values

print("\nMinimum and Maximum Feature Values:")

print(customer_df[
    ['Recency', 'Frequency', 'Monetary', 'AOV']
].agg(['min', 'max']))

# Phase 6.13.3: Detect outliers using IQR

features = ['Recency', 'Frequency', 'Monetary', 'AOV']

for col in features:
    Q1 = customer_df[col].quantile(0.25)
    Q3 = customer_df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = customer_df[
        (customer_df[col] < lower) |
        (customer_df[col] > upper)
    ]

    print(f"\n{col}:")
    print("Lower limit:", round(lower, 2))
    print("Upper limit:", round(upper, 2))
    print("Number of outliers:", len(outliers))


    
# Phase 6.13.5.1: Compare RobustScaler with StandardScaler
from sklearn.preprocessing import RobustScaler

robust_scaler = RobustScaler()
X_robust = robust_scaler.fit_transform(X)

robust_kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

robust_kmeans.fit(X_robust)

customer_df['Robust_Cluster'] = robust_kmeans.labels_

print("\nCluster Counts Using StandardScaler:")
print(customer_df['Cluster'].value_counts().sort_index())

print("\nCluster Counts Using RobustScaler:")
print(customer_df['Robust_Cluster'].value_counts().sort_index())    



# Phase 6.13.5.2: Compare cluster characteristics
robust_summary = customer_df.groupby('Robust_Cluster')[
    ['Recency', 'Frequency', 'Monetary', 'AOV']
].mean()

print("\nRobustScaler Cluster Summary:")
print(robust_summary.round(2))
    

# Phase 6.13.5.3: Compare both scaling methods
comparison = pd.DataFrame({
    'StandardScaler': customer_df['Cluster'].value_counts().sort_index(),
    'RobustScaler': customer_df['Robust_Cluster'].value_counts().sort_index()
})
print("\nCluster Size Comparison:")
print(comparison)  


# Phase 6.14.2: Final Dataset Validation
final_df = pd.read_csv("data/processed/customer_segments.csv")

print("\nFinal Dataset Shape:")
print(final_df.shape)

print("\nFinal Dataset Columns:")
print(final_df.columns.tolist())

print("\nFinal Segment Distribution:")
print(final_df['Segment'].value_counts())
    