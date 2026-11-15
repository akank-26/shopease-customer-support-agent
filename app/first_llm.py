
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# 1. Load configuration from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is missing")

# 2. Initialize the model
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    timeout=30,
    max_retries=2,
)

# 3. Prepare messages
messages = [
    SystemMessage(
        content=(
            "You are a customer support assistant for ShopEase. "
            "You currently have NO access to order databases, "
            "tracking systems, or company policy documents. "
            "Never invent order details or policies. "
            "Never claim you can look up an order. "
            "Do not request personal contact information "
            "because no identity verification tool exists yet. "
            "If asked about an order, explain the limitation "
            "and suggest contacting customer support."
        )
    ),
    HumanMessage(
        content="Where is my order ORD-1024?"
    ),
]

# 4. Measure request latency
start = time.perf_counter()

response = llm.invoke(messages)

latency = time.perf_counter() - start

# 5. Inspect response and metadata
print("\n--- AI RESPONSE ---")
print(response.content)

print("\n--- MODEL METADATA ---")
print("Model:", response.response_metadata.get("model_name"))
print("Finish reason:", response.response_metadata.get("finish_reason"))

usage = response.usage_metadata or {}

print("\n--- TOKEN USAGE ---")
print("Input tokens:", usage.get("input_tokens"))
print("Output tokens:", usage.get("output_tokens"))
print("Total tokens:", usage.get("total_tokens"))

print("\n--- PERFORMANCE ---")
print(f"Latency: {latency:.2f} seconds")
