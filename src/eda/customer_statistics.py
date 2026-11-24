from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_customer_features():
    """
    Load the customer-level feature dataset.
    """

    file_path = PROJECT_ROOT / "data" / "processed" / "customer_features.csv"

    df = pd.read_csv(file_path)

    return df

def calculate_basic_statistics(df):
    """
    Calculate basic descriptive statistics
    for customer behavior features.
    """

    statistics = df[["Recency", "Frequency", "Monetary", "AOV"]].describe()

    return statistics

def plot_feature_distributions(df):
    """
    Plot histograms for customer behavior features.
    """

    features = ["Recency", "Frequency", "Monetary", "AOV"]

    for feature in features:

        plt.figure(figsize=(8, 5))

        plt.hist(df[feature], bins=30)

        plt.title(f"Distribution of {feature}")

        plt.xlabel(feature)

        plt.ylabel("Number of Customers")

        plt.show()


def identify_outliers(df):
    """
    Identify potential outliers using the IQR method.
    """

    features = ["Recency", "Frequency", "Monetary", "AOV"]

    outlier_summary = {}

    for feature in features:

        Q1 = df[feature].quantile(0.25)

        Q3 = df[feature].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR

        upper_bound = Q3 + 1.5 * IQR

        outliers = df[
            (df[feature] < lower_bound) |
            (df[feature] > upper_bound)
        ]

        outlier_summary[feature] = len(outliers)

    return outlier_summary

def plot_outliers(df):
    """
    Create boxplots for customer behavior features.
    """

    features = ["Recency", "Frequency", "Monetary", "AOV"]

    for feature in features:

        plt.figure(figsize=(8, 5))

        plt.boxplot(df[feature])

        plt.title(f"Boxplot of {feature}")

        plt.ylabel(feature)

        plt.show()

def calculate_correlation(df):
    """
    Calculate correlations between customer behavior features.
    """

    features = ["Recency", "Frequency", "Monetary", "AOV"]

    correlation_matrix = df[features].corr()

    return correlation_matrix

def plot_correlation_heatmap(correlation_matrix):
    """
    Plot a heatmap of the customer feature correlations.
    """

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f"
    )

    plt.title("Correlation Heatmap of Customer Features")

    plt.show()




if __name__ == "__main__":

    df = load_customer_features()

    print("Customer features loaded successfully!")

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 customers:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    statistics = calculate_basic_statistics(df)

    print("\nBasic statistics:")
    print(statistics)

    plot_feature_distributions(df)

    outlier_summary = identify_outliers(df)

    print("\nOutlier summary:")

    for feature, count in outlier_summary.items():
        print(f"{feature}: {count} potential outliers")

    plot_outliers(df)

    correlation_matrix = calculate_correlation(df)

    print("\nCorrelation matrix:")

    print(correlation_matrix)

    plot_correlation_heatmap(correlation_matrix)








