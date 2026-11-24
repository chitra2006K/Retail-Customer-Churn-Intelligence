import pandas as pd
from pathlib import Path
from data_cleaning import clean_data

# Find the main project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Location of our original dataset
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Online Retail.xlsx"


def load_data():
    """Load the retail dataset from the Excel file."""
    df = pd.read_excel(DATA_PATH)
    return df


if __name__ == "__main__":
    df = load_data()

    df_clean = clean_data(df)

    print("\nOriginal shape:")
    print(df.shape)

    print("\nCleaned shape:")
    print(df_clean.shape)

    print("\nNegative quantities after cleaning:")
    print((df_clean["Quantity"] < 0).sum())

    print("Dataset loaded successfully!")

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nRows with missing CustomerID:")
    print(df[df["CustomerID"].isnull()].head())

    print("\nRows with missing CustomerID - shape:")
    print(df[df["CustomerID"].isnull()].shape)

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nNegative quantities:")
    print((df["Quantity"] < 0).sum())

    print("\nZero quantities:")
    print((df["Quantity"] == 0).sum())

    print("\nZero or negative UnitPrice:")
    print((df["UnitPrice"] <= 0).sum())

    print("\nInvoices starting with C:")
    print(df["InvoiceNo"].astype(str).str.startswith("C").sum())

    print("\nBasic statistics:")
    print(df.describe(include="all"))

    # Save the cleaned dataset
    output_path = PROJECT_ROOT / "data" / "processed" / "cleaned_retail_data.csv"

    df_clean.to_csv(output_path, index=False)

    print("\nCleaned dataset saved successfully!")
    print(f"Saved to: {output_path}")