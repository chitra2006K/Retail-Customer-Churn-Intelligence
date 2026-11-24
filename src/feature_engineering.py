import pandas as pd
from pathlib import Path


# Find the main project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Location of our cleaned dataset
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_retail_data.csv"


def load_cleaned_data():
    """Load the cleaned retail dataset."""
    df = pd.read_csv(DATA_PATH)

    # Convert InvoiceDate back to datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    return df


if __name__ == "__main__":
    df = load_cleaned_data()

    # Calculate total amount for each transaction
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    # Calculate total spending for each customer
    monetary = df.groupby("CustomerID")["TotalAmount"].sum()

    # Calculate the number of unique orders for each customer
    frequency = df.groupby("CustomerID")["InvoiceNo"].nunique()

    # Find the last purchase date for each customer
    last_purchase = df.groupby("CustomerID")["InvoiceDate"].max()

    # Use the day after the last transaction as the reference date
    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    # Calculate recency in days for each customer
    recency = (reference_date - last_purchase).dt.days

    # Calculate average order value for each customer
    aov = monetary / frequency

    # Combine all customer-level features into one DataFrame
    customer_features = pd.DataFrame({
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary,
        "AOV": aov
    })

    customer_features = customer_features.reset_index()

    print("Cleaned dataset loaded successfully!")

    print("\nShape:")
    print(df.shape)

    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nTotalAmount calculation:")
    print(df[["Quantity", "UnitPrice", "TotalAmount"]].head())

    print("\nMonetary value:")
    print(monetary.head())

    print("\nFrequency:")
    print(frequency.head())

    print("\nReference date:")
    print(reference_date)

    print("\nLast purchase date:")
    print(last_purchase.head())

    print("\nRecency:")
    print(recency.head())

    print("\nAverage Order Value:")
    print(aov.head())

    print("\nCustomer-level features:")
    print(customer_features.head())

    print("\nCustomer-level dataset shape:")
    print(customer_features.shape)

    print("\nCustomer-level data types:")
    print(customer_features.dtypes)

    # Save the customer-level features
    output_path = PROJECT_ROOT / "data" / "processed" / "customer_features.csv"

    customer_features.to_csv(output_path, index=False)

    print("\nCustomer features saved successfully!")
    print(f"Saved to: {output_path}")