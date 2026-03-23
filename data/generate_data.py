"""Generate synthetic fraud transaction data and save to CSV."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import generate_fraud_data

DATA_DIR = os.path.dirname(__file__)


def main():
    df = generate_fraud_data(n_samples=10000, fraud_rate=0.02, random_state=42)

    out_path = os.path.join(DATA_DIR, "fraud_transactions.csv")
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} transactions to {out_path}")
    print(f"Fraud rate: {df['is_fraud'].mean():.4f} ({df['is_fraud'].sum()} fraudulent)")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
