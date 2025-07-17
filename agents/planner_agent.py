# agents/planner_agent.py

import pandas as pd
from utils.llm_utils import call_llm_plan_agent

def planner_agent(df: pd.DataFrame) -> dict:
    print("\U0001F916 Planner Agent (Groq): Analyzing dataset schema...")
    schema = df.dtypes.astype(str).to_dict()
    plan = call_llm_plan_agent(schema)
    print("\u2705 Groq Planner Agent Output:")
    print(plan)
    return plan
