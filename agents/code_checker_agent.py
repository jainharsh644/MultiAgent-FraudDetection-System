# agents/code_checker_agent.py

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def code_checker_agent(code: str):
    print("\U0001f9d1‍\U0001f527 Code Checker Agent: Reviewing modeling agent code using Groq...")

    prompt = f"""
You are a Python code reviewer with expertise in machine learning and fraud analytics.

Please review the following code for:
- Efficiency
- Clarity
- Best practices
- Suggestions for improvement

Only return your review and suggested improvements. Don't include the original code.

```python
{code}
```
"""

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        print("✅ Groq Code Review Suggestions:\n")
        print(response.choices[0].message.content.strip())

    except Exception as e:
        print(f"❌ Groq Code Review failed.\nError: {e}")