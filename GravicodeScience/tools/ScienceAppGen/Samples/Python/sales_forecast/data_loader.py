"""Loading and preparing the daily sales table."""
import pandas as pd

DATA_PATH = "data/sales.csv"


def load_sales(path=DATA_PATH):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.dayofweek
    return df


def summarize(df):
    """Mean daily sales per store, highest first."""
    return df.groupby("store")["sales"].mean().sort_values(ascending=False)
