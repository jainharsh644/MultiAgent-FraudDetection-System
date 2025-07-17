# agents/explainability_agent.py

import shap
import pandas as pd
import matplotlib.pyplot as plt
import os

def explainability_agent(model, X):
    print("\U0001f50d Explainability Agent: Calculating SHAP values...")

    # Defensive check
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")

    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        os.makedirs("outputs", exist_ok=True)

        # SHAP summary plot
        plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        plt.savefig("outputs/shap_summary.png")
        print("✅ SHAP summary plot saved to outputs/shap_summary.png")

    except Exception as e:
        print(f"❌ SHAP explainability failed: {e}")