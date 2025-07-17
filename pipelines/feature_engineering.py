# pipelines/feature_engineering.py

import pandas as pd

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print("🧪 Feature Engineering: Creating temporal and customer-level features...")

    df = df.copy()

    # Ensure correct types
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df.sort_values(by=["customer", "step"], inplace=True)

    # Rolling transaction count per customer over last 24h (approx 5 steps)
    df["rolling_txn_count_24h"] = (
        df.groupby("customer")["step"]
        .rolling(window=5, min_periods=1)
        .count()
        .reset_index(drop=True)
    )

    # Rolling amount over last 24h
    df["rolling_amount_sum_24h"] = (
        df.groupby("customer")["amount"]
        .rolling(window=5, min_periods=1)
        .sum()
        .reset_index(drop=True)
    )

    # Average amount per customer so far
    df["avg_amt_per_customer"] = (
        df.groupby("customer")["amount"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Unique merchant count so far
    # Create feature: number of unique merchants seen so far by each customer
    df["unique_merchants_so_far"] = 0

    # Precompute for each customer
    for cust_id, group in df.groupby("customer"):
        unique_merchants = set()
        counts = []
        for merchant in group["merchant"]:
            unique_merchants.add(merchant)
            counts.append(len(unique_merchants))
        df.loc[group.index, "unique_merchants_so_far"] = counts



    # Risky category indicator
    risky_categories = ["travel", "misc_net", "shopping_net", "entertainment"]
    df["is_risky_category"] = df["category"].isin(risky_categories).astype(int)

    df.fillna(0, inplace=True)
    return df
