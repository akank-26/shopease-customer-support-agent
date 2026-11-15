
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY missing")

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    timeout=30,
    max_retries=2,
)

system_prompt = """
You are ShopEase's customer support assistant.

Current capabilities:
- You can answer general questions.
- You cannot access order databases.
- You cannot retrieve company policy documents.
- You cannot cancel orders or issue refunds.

Rules:
- Never invent order details or company policies.
- Never claim to perform unavailable actions.
- Do not request personal information.
- Explain limitations clearly.
- Suggest human support when necessary.
"""

test_cases = [
    {
        "id": "T01",
        "question": "Where is order ORD-1024?",
    },
    {
        "id": "T02",
        "question": "What is your refund policy?",
    },
    {
        "id": "T03",
        "question": "Can you cancel my order?",
    },
]

for case in test_cases:
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=case["question"]),
        ],
        config={"run_name": f"prompt_eval_{case['id']}"},
    )

    usage = response.usage_metadata or {}

    print(f"\n--- {case['id']} ---")
    print("Question:", case["question"])
    print("Response:", response.content)
    print("Tokens:", usage.get("total_tokens"))
