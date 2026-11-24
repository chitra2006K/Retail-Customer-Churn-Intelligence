import pandas as pd


def clean_data(df):
    """
    Clean the retail transaction dataset.

    Cleaning steps:
    1. Remove rows without CustomerID.
    2. Remove exact duplicate rows.
    3. Remove cancelled invoices.
    4. Remove rows with non-positive UnitPrice.
    """

    # Make a copy so the original DataFrame is not modified unexpectedly
    df = df.copy()

    # 1. Remove transactions without a CustomerID
    df = df.dropna(subset=["CustomerID"])

    # 2. Remove exact duplicate rows
    df = df.drop_duplicates()

    # 3. Remove cancelled invoices
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # 4. Keep only transactions with a positive UnitPrice
    df = df[df["UnitPrice"] > 0]

    return df