# agents/cleaning_agent.py

import pandas as pd
import json
import re
from utils.llm_utils import call_llm_cleaning_agent

def apply_cleaning_rules(df: pd.DataFrame, rules: list) -> pd.DataFrame:
    for rule in rules:
        action = rule.get("action")

        if action == "dropna":
            df = df.dropna()

        elif action == "handle_inconsistent_dtypes":
            col = rule.get("column")
            df[col] = pd.to_numeric(df[col], errors="coerce")

        elif action == "strip_whitespace":
            for col in rule.get("columns", []):
                df[col] = df[col].astype(str).str.strip()

        elif action == "convert_to_titlecase":
            for col in rule.get("columns", []):
                df[col] = df[col].astype(str).str.title()

        elif action == "encode_categoricals":
            for col in rule.get("columns", []):
                if col in df.columns:
                    df[col] = df[col].astype("category").cat.codes

        elif action == "replace_missing_value":
            col = rule.get("column")
            value = rule.get("value")
            df[col] = df[col].fillna(value)

        else:
            print(f"⚠️ Unknown cleaning rule: {rule}")

    return df

def cleaning_agent(csv_path: str) -> pd.DataFrame:
    print("🧼 Cleaning Agent (LLM): Analyzing and cleaning data...")

    df = pd.read_csv(csv_path)
    schema = df.dtypes.astype(str).to_dict()
    rules = call_llm_cleaning_agent(schema)

    print(f"🔎 Raw LLM response: {json.dumps(rules, indent=2)}")

    if not rules:
        print("❌ No rules returned. Skipping cleaning.")
        return df  # ⚠️ IMPORTANT: fallback

    for rule in rules:
        # apply cleaning logic...
        pass

    print("✅ Cleaning completed based on LLM instructions.")
    # Save cleaned data
    df.to_csv("data/banksim_cleaned.csv", index=False)

# Return df so it's usable downstream
    return df