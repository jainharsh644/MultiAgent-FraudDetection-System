# agents/memory_agent.py

import json
import os

MEMORY_PATH = "outputs/memory.json"

def memory_agent(action: str, memory: dict = None) -> dict:
    """
    Handles loading or saving memory between agents.

    Args:
        action (str): "load" or "save"
        memory (dict): memory to save if action is "save"

    Returns:
        dict: loaded memory if action is "load"
    """
    os.makedirs("outputs", exist_ok=True)

    if action == "load":
        if os.path.exists(MEMORY_PATH):
            try:
                with open(MEMORY_PATH, "r") as f:
                    print("📦 Memory loaded from previous run.")
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load memory: {e}")
        return {}

    elif action == "save":
        if memory is None:
            raise ValueError("Memory dict must be provided for save action.")
        try:
            with open(MEMORY_PATH, "w") as f:
                json.dump(memory, f, indent=2)
                print("💾 Memory saved to outputs/memory.json.")
        except Exception as e:
            print(f"❌ Failed to save memory: {e}")
    else:
        raise ValueError("Invalid action. Use 'load' or 'save'.")
