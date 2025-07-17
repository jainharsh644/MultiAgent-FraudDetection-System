# utils/llm_utils.py

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"  # Groq's endpoint
)


def call_llm_cleaning_agent(schema: dict) -> list:
    """
    Calls Groq LLM to suggest cleaning steps based on schema.
    Returns a list of cleaning rules in dict format.
    """
    prompt = f"""
You are a data cleaning expert. Given this dataset schema as a Python dict of column names and dtypes:
{schema}

Suggest JSON cleaning rules in this format:
[
  {{"action": "dropna"}},
  {{"action": "handle_inconsistent_dtypes", "column": "age"}},
  {{"action": "strip_whitespace", "columns": ["customer", "gender", "merchant", "category"]}},
  {{"action": "convert_to_titlecase", "columns": ["customer", "gender", "merchant", "category"]}},
  {{"action": "encode_categoricals", "columns": ["gender", "category", "merchant"]}}
]

Only return valid JSON. No commentary.
"""
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"❌ Cleaning LLM failed: {e}")
        return []


def call_llm_plan_agent(schema: dict) -> dict:
    """
    Calls Groq LLM to return a pipeline plan: features, model, scaling, imbalance_strategy.
    """
    prompt = f"""
You are a fraud analytics pipeline planner. Given this dataset schema:
{schema}

Suggest a JSON plan:
{{
  "features": [...],
  "model": "xgboost | logistic_regression | random_forest | lightgbm",
  "scaling": "standard | minmax | none",
  "imbalance_strategy": "scale_pos_weight | smote | undersample | none"
}}

Only return valid JSON. No explanation.
"""
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"❌ Planner LLM failed: {e}")
        return {
            "features": ["amount", "age", "category"],
            "model": "xgboost",
            "scaling": "standard",
            "imbalance_strategy": "scale_pos_weight"
        }


def call_llm_rule_suggester(top_features: list, feature_names: list) -> str:
    """
    Calls Groq LLM to suggest rule-based fraud detection logic.
    Returns a Python code string.
    """
    prompt = f"""
You are an expert fraud analyst. Based on the following top features (most important for fraud detection):
{top_features}

And the complete list of available features:
{feature_names}

Suggest 3 if-else rule-based heuristics to catch fraud. Output Python code only using 'if' conditions on features and a final `rules_output` Series of 0/1 predictions.

Return a Python code block like this:

import pandas as pd
def rule_based_flags(df):
    rules_output = pd.Series(0, index=df.index)

    # Rule 1
    rules_output |= (df['amount'] > 1000) & (df['category'] == 'travel')

    # Rule 2
    rules_output |= (df['age'] < 18) & (df['merchant'] == 'crypto')

    # Rule 3
    rules_output |= (df['rolling_txn_count_24h'] > 10)

    return rules_output
"""
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"❌ Rule suggestion LLM failed: {e}")
        return ""
