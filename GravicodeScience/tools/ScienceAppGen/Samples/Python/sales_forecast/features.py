"""Turning the sales table into a feature matrix."""
import numpy as np
from sklearn.preprocessing import StandardScaler

FEATURES = ["price", "promo", "month", "weekday", "temperature"]


def build_features(df):
    x = df[FEATURES].to_numpy(dtype=float)
    y = df["sales"].to_numpy(dtype=float)
    return x, y


def scale(train_x, test_x):
    scaler = StandardScaler()
    return scaler.fit_transform(train_x), scaler.transform(test_x)


def weekend_share(df):
    weekend = df["weekday"] >= 5
    return float(np.mean(weekend))
