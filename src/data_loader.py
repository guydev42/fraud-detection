"""
Synthetic credit card fraud data generator and loader.
Produces 10K transactions with ~2% fraud rate and realistic feature distributions.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_fraud_data(n_samples=10000, fraud_rate=0.02, random_state=42):
    """
    Generate synthetic credit card transaction data.

    Features:
        amount              - transaction amount in dollars
        time_hour           - hour of day (0-23)
        merchant_category   - encoded merchant type (0-9)
        distance_from_home  - miles from cardholder's home
        distance_from_last_transaction - miles from previous transaction
        ratio_to_median_purchase - amount / median purchase for this card
        is_weekend          - 1 if Saturday or Sunday
        is_night            - 1 if between 22:00 and 06:00
        num_transactions_last_hour - transaction velocity (1h window)
        num_transactions_last_day  - transaction velocity (24h window)

    Target:
        is_fraud - binary (0 = legitimate, 1 = fraud)

    Returns:
        pd.DataFrame with all features and target
    """
    rng = np.random.RandomState(random_state)

    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    # --- Legitimate transactions ---
    legit = {
        "amount": rng.lognormal(mean=3.5, sigma=1.0, size=n_legit).clip(0.50, 5000),
        "time_hour": rng.choice(24, size=n_legit, p=_hour_weights_legit()),
        "merchant_category": rng.choice(10, size=n_legit, p=_merchant_weights_legit()),
        "distance_from_home": rng.exponential(scale=8.0, size=n_legit).clip(0, 200),
        "distance_from_last_transaction": rng.exponential(scale=5.0, size=n_legit).clip(0, 150),
        "ratio_to_median_purchase": rng.lognormal(mean=0.0, sigma=0.4, size=n_legit).clip(0.1, 10),
        "is_weekend": rng.binomial(1, 0.28, size=n_legit),
        "is_night": rng.binomial(1, 0.10, size=n_legit),
        "num_transactions_last_hour": rng.poisson(lam=0.8, size=n_legit),
        "num_transactions_last_day": rng.poisson(lam=5.0, size=n_legit),
        "is_fraud": np.zeros(n_legit, dtype=int),
    }

    # --- Fraudulent transactions ---
    fraud = {
        "amount": rng.lognormal(mean=5.0, sigma=1.5, size=n_fraud).clip(1, 25000),
        "time_hour": rng.choice(24, size=n_fraud, p=_hour_weights_fraud()),
        "merchant_category": rng.choice(10, size=n_fraud, p=_merchant_weights_fraud()),
        "distance_from_home": rng.exponential(scale=50.0, size=n_fraud).clip(0, 500),
        "distance_from_last_transaction": rng.exponential(scale=40.0, size=n_fraud).clip(0, 500),
        "ratio_to_median_purchase": rng.lognormal(mean=1.5, sigma=0.8, size=n_fraud).clip(0.5, 50),
        "is_weekend": rng.binomial(1, 0.35, size=n_fraud),
        "is_night": rng.binomial(1, 0.45, size=n_fraud),
        "num_transactions_last_hour": rng.poisson(lam=3.0, size=n_fraud),
        "num_transactions_last_day": rng.poisson(lam=12.0, size=n_fraud),
        "is_fraud": np.ones(n_fraud, dtype=int),
    }

    df_legit = pd.DataFrame(legit)
    df_fraud = pd.DataFrame(fraud)
    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Round amounts for realism
    df["amount"] = df["amount"].round(2)
    df["distance_from_home"] = df["distance_from_home"].round(2)
    df["distance_from_last_transaction"] = df["distance_from_last_transaction"].round(2)
    df["ratio_to_median_purchase"] = df["ratio_to_median_purchase"].round(3)

    return df


def _hour_weights_legit():
    """Probability of transaction by hour for legitimate transactions."""
    w = np.array([
        1, 0.5, 0.3, 0.2, 0.2, 0.3, 1, 2, 4, 5, 5, 6,
        7, 6, 5, 5, 5, 6, 6, 5, 4, 3, 2, 1.5,
    ], dtype=float)
    return w / w.sum()


def _hour_weights_fraud():
    """Probability of transaction by hour for fraud (skewed toward night)."""
    w = np.array([
        5, 5, 6, 6, 5, 4, 2, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 5,
    ], dtype=float)
    return w / w.sum()


def _merchant_weights_legit():
    """Distribution over 10 merchant categories for legitimate transactions."""
    w = np.array([15, 12, 10, 10, 10, 10, 8, 8, 9, 8], dtype=float)
    return w / w.sum()


def _merchant_weights_fraud():
    """Distribution over 10 merchant categories for fraud (concentrated)."""
    w = np.array([3, 3, 5, 15, 5, 3, 20, 18, 15, 13], dtype=float)
    return w / w.sum()


def load_and_prepare(filepath="data/fraud_transactions.csv", test_size=0.2, random_state=42):
    """
    Load fraud transaction data and return train/test splits.

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    df = pd.read_csv(filepath)

    feature_cols = [c for c in df.columns if c != "is_fraud"]
    X = df[feature_cols].values.astype(float)
    y = df["is_fraud"].values
    feature_names = list(feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set:     {X_test.shape[0]} samples")
    print(f"Features:     {X_train.shape[1]}")
    print(f"Fraud rate (train): {y_train.mean():.4f}")
    print(f"Fraud rate (test):  {y_test.mean():.4f}")

    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    df = generate_fraud_data()
    print(f"Generated {len(df)} transactions")
    print(f"Fraud rate: {df['is_fraud'].mean():.4f}")
    print(f"\nFeature summary:\n{df.describe().round(2)}")
