
import pytest

import app.combined_agent as agent


@pytest.mark.parametrize("question", [
    "Can you cancel order ORD-1024?",
    "Please cancel my order ORD-1024",
    "Cancel order ORD-1025",
])
def test_cancellation_is_refused_without_ai_calls(
    monkeypatch, question
):
    def unexpected_retrieval(*args, **kwargs):
        raise AssertionError("Retrieval should not run")

    monkeypatch.setattr(
        agent,
        "search_documents",
        unexpected_retrieval,
    )

    result = agent.answer_customer(
        question=question,
        authenticated_customer_id="CUST-001",
    )

    assert "can't cancel orders" in result["answer"]
    assert "No cancellation has been performed" in result["answer"]
    assert result["tool_used"] is False


def test_policy_question_not_classified_as_cancellation():
    assert not agent.is_cancellation_request(
        "What is the order cancellation policy?"
    )
