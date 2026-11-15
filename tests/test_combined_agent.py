import pytest

from app.combined_agent import answer_customer


def test_empty_question():
    with pytest.raises(ValueError):
        answer_customer(
            question="",
            authenticated_customer_id="CUST-001",
        )