
"""ShopEase offline evaluation for R01–R06.

Run from the customer-support-agent project root:

    python offline_eval.py --self-test
    python offline_eval.py

The evaluators use deterministic checks for known requirements.
They do not guarantee complete semantic correctness or safety.
"""

import argparse
import os
import re
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATASET_NAME = os.getenv(
    "LANGSMITH_EVAL_DATASET",
    "shopease-support-golden-set",
)


# The six questions in our golden dataset.
QUESTIONS = {
    "R01": "Can I return unused headphones after 20 days?",
    "R02": "How long does an approved refund take?",
    "R03": "Can I return a digital product?",
    "R04": "What is ShopEase's customer support phone number?",
    "R05": "Can you cancel my order immediately?",
    "R06": "Am I eligible for a refund if my package has not arrived?",
}

QUESTION_IDS = {
    question: case_id
    for case_id, question in QUESTIONS.items()
}


def has(text: str, pattern: str) -> bool:
    """Check whether a regex pattern matches some text."""
    return re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    ) is not None


# --------------------------------------------------
# 1. Run the actual Customer Support Agent
# --------------------------------------------------

def run_agent(inputs: dict) -> dict:
    """Execute one golden question using our existing agent."""

    from app.combined_agent import answer_customer

    question = inputs.get("question")

    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            "Dataset example needs a nonempty 'question'"
        )

    customer_id = inputs.get(
        "authenticated_customer_id",
        "CUST-001",
    )

    if not isinstance(customer_id, str) or not customer_id.strip():
        raise ValueError(
            "Dataset example needs authenticated_customer_id"
        )

    k = inputs.get("k", 4)

    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(
            "Dataset example 'k' must be a positive integer"
        )

    return answer_customer(
        question=question,
        authenticated_customer_id=customer_id,
        k=k,
    )


# --------------------------------------------------
# 2. Evaluator: Check for a nonempty answer
# --------------------------------------------------

def answer_nonempty(outputs: dict) -> dict:
    """Check whether the agent returned an answer."""

    answer = outputs.get("answer")

    passed = (
        isinstance(answer, str)
        and bool(answer.strip())
    )

    return {
        "key": "answer_nonempty",
        "score": int(passed),
        "comment": (
            "Nonempty answer"
            if passed
            else "Answer missing or blank"
        ),
    }


# --------------------------------------------------
# 3. Evaluator: Check golden-case requirements
# --------------------------------------------------

def case_requirements(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    """Check known requirements for each R01–R06 question."""

    question = str(inputs.get("question", ""))
    answer = str(outputs.get("answer") or "")

    case_id = QUESTION_IDS.get(question)
    reference_outputs = reference_outputs or {}

    if case_id is None:
        return {
            "key": "golden_case_requirements",
            "score": 0,
            "comment": (
                "Unknown dataset question. "
                "Add a dedicated evaluation rule."
            ),
        }

    checks = {}

    # R01 — Return unused headphones.
    if case_id == "R01":
        checks = {
            "30-day return window": has(
                answer,
                r"\b(?:30|thirty)\s*(?:-|\s)*days?\b",
            ),
            "unused condition": has(
                answer,
                r"\b(?:unused|not\s+used|unopened)\b",
            ),
            "original packaging": has(
                answer,
                r"\boriginal\s+(?:packaging|package|box)\b",
            ),
            "valid order ID": has(
                answer,
                r"\b(?:valid\s+)?order\s*(?:id|number)\b",
            ),
            "qualified, not guaranteed": has(
                answer,
                r"\b(?:if|provided|subject to|depends|need|require|eligible when|eligibility)\b",
            ),
        }

    # R02 — Approved refund timeline.
    elif case_id == "R02":
        checks = {
            "5–7 business days": has(
                answer,
                r"\b5\s*(?:-|–|to)\s*7\s+business\s+days\b",
            ),
            "after inspection": has(
                answer,
                r"\b(?:after|following)\s+(?:(?:the|a)\s+)?(?:returned\s+item\s+)?(?:passes\s+)?inspection\b|\bafter\s+(?:the\s+)?(?:item\s+)?(?:is\s+)?inspected\b",
            ),
            "original payment method": has(
                answer,
                r"\boriginal\s+payment\s+method\b|\bsame\s+payment\s+method\b",
            ),
        }

    # R03 — Digital products cannot be returned.
    elif case_id == "R03":
        checks = {
            "digital product excluded": (
                has(answer, r"\bdigital\b")
                and has(
                    answer,
                    r"\b(?:non[ -]?returnable|cannot\s+be\s+returned|can'?t\s+return|not\s+(?:eligible|returnable)|no\s+returns?)\b",
                )
            ),
        }

    # R04 — Do not invent a customer support phone number.
    elif case_id == "R04":
        phone_number = has(
            answer,
            r"(?<!\d)(?:\+?\d[\s().-]*){7,15}(?!\d)",
        )

        checks = {
            "no phone number invented": not phone_number,
            "acknowledges unavailable contact info": has(
                answer,
                r"\b(?:don'?t|do\s+not|cannot|can'?t|unable|not\s+(?:available|provided|listed|verified)|no\s+(?:verified|official|phone|number)|contact\s+(?:details|information)\s+unavailable)\b",
            ),
        }

    # R05 — AI cannot cancel customer orders.
    elif case_id == "R05":
        checks = {
            "does not claim cancellation performed": not has(
                answer,
                r"\b(?:i\s+(?:have|already)\s+cancelled|your\s+order\s+(?:has\s+been|is)\s+cancelled|successfully\s+cancelled)\b",
            ),
            "declines cancellation": has(
                answer,
                r"\b(?:can'?t|cannot|unable|not\s+able|don'?t\s+have\s+(?:the\s+)?ability|not\s+supported|no\s+cancellation\s+has\s+been\s+performed)\b",
            ),
            "human support escalation": has(
                answer,
                r"\b(?:human\s+support|support\s+(?:team|representative|agent)|contact\s+support|customer\s+service)\b",
            ),
        }

    # R06 — Undelivered-order refund eligibility.
    elif case_id == "R06":
        checks = {
            "10-day shipment condition": (
                has(
                    answer,
                    r"\b(?:10|ten)\s+(?:calendar\s+)?days?\b",
                )
                and has(
                    answer,
                    r"\bship(?:ped|ment|ping)?\b",
                )
            ),
            "not delivered condition": has(
                answer,
                r"\b(?:not\s+(?:been\s+)?delivered|undelivered|hasn'?t\s+(?:been\s+)?(?:arrived|delivered)|not\s+arrived)\b",
            ),
            "no active delivery exception": has(
                answer,
                r"\b(?:no|without|not\s+have|mustn'?t\s+have)\s+(?:an?\s+)?active\s+delivery\s+exception\b|\bdelivery\s+exception\b",
            ),
            "does not guarantee refund": not has(
                answer,
                r"\b(?:you\s+are\s+(?:definitely\s+)?eligible|you\s+will\s+(?:definitely\s+)?get\s+a\s+refund|refund\s+(?:is|has\s+been)\s+approved)\b",
            ),
        }

    missing = [
        label
        for label, passed in checks.items()
        if not passed
    ]

    expected_source = reference_outputs.get(
        "expected_source",
        "not specified",
    )

    if missing:
        comment = (
            f"{case_id}; expected source={expected_source}. "
            f"Missing/failed checks: {', '.join(missing)}."
        )
    else:
        comment = (
            f"{case_id}; expected source={expected_source}. "
            f"All {len(checks)} narrow checks passed."
        )

    return {
        "key": "golden_case_requirements",
        "score": 0 if missing else 1,
        "comment": (
            comment
            + " Heuristic only; review semantic correctness "
            "and supporting sources separately."
        ),
    }


# --------------------------------------------------
# 4. Self-test without API calls
# --------------------------------------------------

def self_test() -> None:
    """Test evaluation rules using mock answers."""

    examples = {
        "R01": (
            "Unused headphones can be returned within "
            "30 days from delivery if in original packaging "
            "and you provide a valid order ID. "
            "Eligibility must be verified."
        ),
        "R02": (
            "Approved refunds go to the original payment "
            "method 5–7 business days after inspection."
        ),
        "R03": "Digital products are non-returnable.",
        "R04": (
            "I don't have a verified customer support "
            "phone number to share."
        ),
        "R05": (
            "I can't cancel orders directly. "
            "Human support is required."
        ),
        "R06": (
            "Refund may qualify after at least 10 days "
            "since shipment if the order is not delivered "
            "and there is no active delivery exception; "
            "eligibility must be checked."
        ),
    }

    for case_id, answer in examples.items():
        result = case_requirements(
            {"question": QUESTIONS[case_id]},
            {"answer": answer},
            {"expected_source": "mock_source.md"},
        )

        assert result["score"] == 1, (
            case_id,
            result,
        )

        print(f"{case_id} example: PASS")

    # Verify that our earlier defect is detected.
    known_bad = case_requirements(
        {"question": QUESTIONS["R01"]},
        {
            "answer": (
                "Unused headphones can be returned "
                "within 30 days in original packaging."
            )
        },
        {"expected_source": "return_policy.md"},
    )

    assert (
        known_bad["score"] == 0
        and "valid order ID" in known_bad["comment"]
    )

    print(
        "R01 missing-order-ID example: "
        "FAIL detected (expected)"
    )
    print(
        "Local evaluator self-test passed. "
        "No model or LangSmith calls were made."
    )


# --------------------------------------------------
# 5. Run the LangSmith offline experiment
# --------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Test scoring rules without API calls",
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not os.getenv("LANGSMITH_API_KEY"):
        raise SystemExit(
            "LANGSMITH_API_KEY missing from .env"
        )

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY missing from .env"
        )

    from langsmith import Client

    client = Client()

    dataset = client.read_dataset(
        dataset_name=DATASET_NAME,
    )

    print(f"Dataset: {dataset.name}")
    print(
        "Running offline evaluation. "
        "Real model calls may incur costs."
    )

    client.evaluate(
        run_agent,
        data=DATASET_NAME,
        evaluators=[
            answer_nonempty,
            case_requirements,
        ],
        experiment_prefix="shopease-offline-r01-r06",
        description=(
            "Six targeted deterministic regression checks "
            "plus answer_nonempty."
        ),
        max_concurrency=1,
    )

    print(
        "Evaluation complete. "
        "Inspect scores and comments in LangSmith."
    )


if __name__ == "__main__":
    main()
