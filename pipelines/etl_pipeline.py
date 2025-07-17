# pipelines/etl_pipeline.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.class_weight import compute_sample_weight

from pipelines.feature_engineering import apply_feature_engineering

def run_etl_pipeline(plan: dict):
    print("🔄 ETL Pipeline: Cleaning, engineering, and scaling data...")

    # 1. Load cleaned data
    df = pd.read_csv("data/banksim_cleaned.csv")

    # 2. Feature Engineering
    df = apply_feature_engineering(df)

    # 3. Select features as per LLM plan
    selected_features = plan.get("features", [])
    missing = [f for f in selected_features if f not in df.columns]
    if missing:
        print(f"⚠️ Warning: Some features from plan are missing in data: {missing}")
        selected_features = [f for f in selected_features if f in df.columns]

    # 4. Encode any remaining object/categorical columns to numeric (in case LLM agent failed)
    for col in selected_features:
        if df[col].dtype == 'object':
            print(f"⚠️ Encoding column '{col}' as it is still object dtype.")
            df[col] = df[col].astype('category').cat.codes

    X = df[selected_features].copy()
    y = df["fraud"].astype(int)

    # 5. Apply scaling
    scaling = plan.get("scaling", "none").lower()
    if scaling != "none":
        print(f"🧮 Applying {scaling} scaling to numeric features.")
        scaler = StandardScaler() if scaling == "standard" else MinMaxScaler()
        num_cols = X.select_dtypes(include=["float64", "int64"]).columns
        X[num_cols] = scaler.fit_transform(X[num_cols])

    # 6. Handle class imbalance
    imb_strategy = plan.get("imbalance_strategy", "none").lower()
    if imb_strategy == "scale_pos_weight":
        sample_weights = compute_sample_weight(class_weight="balanced", y=y)
    else:
        sample_weights = None

    print(f"✅ Final shape: {X.shape}")
    return X, y, sample_weights
