"""Create/populate the ShopEase LangSmith golden dataset with six RAG cases.

Run from the customer-support-agent project root:
    python create_shopease_golden_dataset.py

Safe to rerun: skips existing examples with the same question.
Requires LANGSMITH_API_KEY in .env or the environment.
Does NOT run the agent or make OpenAI model calls.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATASET_NAME = "shopease-support-golden-set"
DESCRIPTION = (
    "ShopEase Customer Support Agent golden cases covering RAG, policy citations, "
    "refusals, and unanswerable support questions."
)

# These six cases preserve the reference facts/behaviors from rag_golden_dataset.py.
# More examples for authorized order lookup and escalation can be added later.
CASES = [
    {
        "id": "R01",
        "question": "Can I return unused headphones after 20 days?",
        "expected_source": "return_policy.md",
        "reference_facts": [
            "30 days from delivery",
            "Unused and original packaging",
            "Valid order ID required",
        ],
        "expected_behavior": "Explain conditions; do not confirm eligibility without delivery details",
    },
    {
        "id": "R02",
        "question": "How long does an approved refund take?",
        "expected_source": "refund_policy.md",
        "reference_facts": [
            "5–7 business days after inspection",
            "Original payment method",
        ],
        "expected_behavior": "Answer with refund timeline and source",
    },
    {
        "id": "R03",
        "question": "Can I return a digital product?",
        "expected_source": "return_policy.md",
        "reference_facts": ["Digital products are non-returnable"],
        "expected_behavior": "Explain exclusion",
    },
    {
        "id": "R04",
        "question": "What is ShopEase's customer support phone number?",
        "expected_source": "contact_support.md",
        "reference_facts": ["No verified phone number is provided"],
        "expected_behavior": "Do not invent contact information",
    },
    {
        "id": "R05",
        "question": "Can you cancel my order immediately?",
        "expected_source": "contact_support.md",
        "reference_facts": ["AI cannot cancel orders", "Human support required"],
        "expected_behavior": "Decline execution; explain human escalation",
    },
    {
        "id": "R06",
        "question": "Am I eligible for a refund if my package has not arrived?",
        "expected_source": "refund_policy.md",
        "reference_facts": [
            "At least 10 days since shipment",
            "Order not delivered",
            "No active delivery exception",
        ],
        "expected_behavior": "Explain all conditions; do not confirm individual eligibility",
    },
]


def main() -> None:
    if not os.getenv("LANGSMITH_API_KEY"):
        raise SystemExit("LANGSMITH_API_KEY missing: add it to your local .env")

    client = Client()
    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        print(f"Using existing dataset: {dataset.name}")
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=DESCRIPTION,
        )
        print(f"Created dataset: {dataset.name}")

    existing_questions = {
        example.inputs.get("question")
        for example in client.list_examples(dataset_id=dataset.id)
        if isinstance(example.inputs, dict)
    }

    added = 0
    skipped = 0
    for case in CASES:
        if case["question"] in existing_questions:
            print(f"SKIP {case['id']}: question already exists")
            skipped += 1
            continue

        client.create_example(
            dataset_id=dataset.id,
            inputs={
                "question": case["question"],
                "authenticated_customer_id": "CUST-001",  # mock identity
                "k": 4,
            },
            outputs={
                "expected_source": case["expected_source"],
                "reference_facts": case["reference_facts"],
                "expected_behavior": case["expected_behavior"],
            },
            metadata={"case_id": case["id"], "category": "rag_policy"},
            split="base",
        )
        print(f"ADD  {case['id']}: {case['question']}")
        added += 1

    print(f"Done: {added} added, {skipped} skipped (of {len(CASES)} supplied cases).")
    print(f"Open LangSmith → Datasets & Experiments → {DATASET_NAME} → Examples")
    print("Note: reference_facts are evaluation criteria, not a verbatim expected answer.")


if __name__ == "__main__":
    main()
