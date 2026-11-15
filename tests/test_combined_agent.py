
import pytest

from app.combined_agent import answer_customer


def test_empty_question():
    with pytest.raises(ValueError):
        answer_customer("My order ORD-1024 hasn't arrived. Can I get a refund?")
