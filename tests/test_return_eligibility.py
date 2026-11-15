
from app.combined_agent import answer_customer


def test_return_answer_includes_mandatory_order_id():
    result = answer_customer(
        question="Can I return unused headphones?",
        authenticated_customer_id="CUST-001",
    )

    answer = result["answer"].lower()

    assert "order id" in answer or "order number" in answer, (
        "Return-policy answer omitted the mandatory "
        "valid order ID requirement."
    )
