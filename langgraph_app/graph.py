# langgraph_app/graph.py

from agents.cleaning_agent import cleaning_agent
from agents.planner_agent import planner_agent
from agents.modeling_agent import modeling_agent
from agents.rule_suggester_agent import rule_suggester_agent
from agents.explainability_agent import explainability_agent
from agents.code_checker_agent import code_checker_agent
from agents.memory_agent import memory_agent
from pipelines.etl_pipeline import run_etl_pipeline

def run_agentic_pipeline():
    print("🚀 Launching Agentic AI Fraud Detection System...")

    # 0. Cleaning data
    df = cleaning_agent("data/banksim.csv")

    # 1. Load or create memory
    memory = memory_agent("load")

    # 2. Planner agent decides pipeline plan
    plan = planner_agent(df)

    memory["last_plan"] = plan

    # 4. ETL pipeline with plan's features
    X, y, sample_weights  = run_etl_pipeline(plan)
    memory["features_used"] = list(X.columns)

    # 5. Modeling
    model, preds = modeling_agent(plan, X, y)
    memory["model_used"] = plan["model"]

    # 6. Code Review (on modeling_agent.py)
    try:
        with open("agents/modeling_agent.py") as f:
            code_checker_agent(f.read())
    except Exception as e:
        print(f"⚠️ Skipped code check: {e}")

    # 7. Explainability
    explainability_agent(model, X)

    # 8. Rule Suggestions
    rule_suggester_agent(model, X)

    # 9. Save memory
    memory_agent("save", memory)

    print("\n✅ Pipeline completed. All outputs saved in /outputs.")
