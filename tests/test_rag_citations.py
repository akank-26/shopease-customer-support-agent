
import pytest

from langchain_core.documents import Document
from app.rag import prepare_context, extract_citations


def test_citation_mapping():
    doc = Document(
        page_content="Returns within 30 days of delivery.",
        metadata={
            "source": "return_policy.md",
            "chunk_id": "RETURN-001-001",
        },
    )

    context, sources = prepare_context([doc])

    assert "[S1]" in context
    assert sources["S1"]["source"] == "return_policy.md"

    cited = extract_citations(
        "The policy allows 30 days. [S1]",
        sources,
    )

    assert "S1" in cited


def test_invalid_citation_rejected():
    with pytest.raises(ValueError):
        extract_citations(
            "The policy allows 30 days. [S99]",
            {"S1": {"source": "return_policy.md"}},
        )


def test_no_citation_returns_empty_mapping():
    assert extract_citations(
        "I don't have enough information.",
        {"S1": {"source": "return_policy.md"}},
    ) == {}
