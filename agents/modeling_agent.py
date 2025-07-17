# agents/modeling_agent.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
import os

def modeling_agent(plan: dict, X: pd.DataFrame, y: pd.Series):
    print("🤖 Modeling Agent: Building model based on LLM plan...")

    model_name = plan.get("model", "xgboost")
    scaling_method = plan.get("scaling", "standard")
    imbalance_strategy = plan.get("imbalance_strategy", "none")

    # ➤ Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # ➤ Apply scaling
    if scaling_method == "standard":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print("✅ Applied Standard Scaling.")
    elif scaling_method == "minmax":
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print("✅ Applied Min-Max Scaling.")
    else:
        print("⚠️ Skipping feature scaling.")

    # ➤ Handle imbalance
    if imbalance_strategy == "smote":
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print("✅ Applied SMOTE oversampling.")
    elif imbalance_strategy == "undersample":
        rus = RandomUnderSampler(random_state=42)
        X_train, y_train = rus.fit_resample(X_train, y_train)
        print("✅ Applied Random Undersampling.")
    elif imbalance_strategy == "scale_pos_weight" and model_name == "xgboost":
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    else:
        scale_pos_weight = 1

    # ➤ Initialize the model
    if model_name == "xgboost":
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
    elif model_name == "random_forest":
        model = RandomForestClassifier(class_weight='balanced', random_state=42)
    elif model_name == "logistic_regression":
        model = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
    else:
        raise ValueError(f"❌ Unsupported model: {model_name}")

    # ➤ Train model
    model.fit(X_train, y_train)
    print(f"✅ Model trained using {model_name}.")

    # ➤ Evaluate model
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("\n📈 Model Performance:\n")
    print(report)

    # ➤ Save predictions
    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred
    }).to_csv("outputs/fraud_scores.csv", index=False)
    print("✅ Predictions saved to outputs/fraud_scores.csv")

    return model, y_pred
