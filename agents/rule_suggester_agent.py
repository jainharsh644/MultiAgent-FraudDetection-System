# agents/rule_suggester_agent.py

import shap
import pandas as pd
from utils.llm_utils import call_llm_rule_suggester

def rule_suggester_agent(model, X):
    print("\n🤖 Rule Suggester Agent: Asking LLM for fraud rules...")

    # Compute SHAP values
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Get top 5 most impactful features
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": abs(shap_values.values).mean(axis=0)
    }).sort_values(by="importance", ascending=False)

    top_features = feature_importance["feature"].head(5).tolist()

    # Ask LLM for rule suggestions based on top features
    suggestions = call_llm_rule_suggester(top_features, X.columns.tolist())

    # Save suggestions to file
    with open("outputs/rule_suggestions.txt", "w") as f:
        f.write(suggestions)

    print("✅ Saved rule suggestions to outputs/rule_suggestions.txt")
