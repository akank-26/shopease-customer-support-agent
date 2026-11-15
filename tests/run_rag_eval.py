
print("RAG evaluation script started", flush=True)
import json
from pathlib import Path

from app.rag import generate_answer


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "tests" / "rag_golden_set.json"


def evaluate_case(case):
    """Run one RAG question and check basic properties."""

    result = generate_answer(case["question"], k=3)

    answer = result["answer"]
    retrieved = result.get("retrieved_sources", {})
    citations = result.get("citations", {})

    # Check whether the expected document was retrieved.
    retrieved_sources = {
        item["source"]
        for item in retrieved.values()
    }

    source_found = (
        case["expected_source"] in retrieved_sources
    )

    # Check whether all cited labels exist in retrieval.
    citation_ids_valid = all(
        label in retrieved
        for label in citations
    )

    # Check whether the answer actually cites a source.
    has_citations = bool(citations)

    print(f"\n{'=' * 55}")
    print(f"Test: {case['id']}")
    print(f"Question: {case['question']}")

    print("\nEXPECTED FACTS:")
    for fact in case["reference_facts"]:
        print(f"  - {fact}")

    print("\nACTUAL ANSWER:")
    print(answer)

    print("\nAUTOMATED CHECKS:")
    print("Expected source retrieved:", source_found)
    print("Citation IDs valid:", citation_ids_valid)
    print("At least one citation:", has_citations)

    print("\nMANUAL REVIEW:")
    print("Expected behavior:", case["expected_behavior"])

    return {
        "id": case["id"],
        "source_found": source_found,
        "citation_ids_valid": citation_ids_valid,
        "has_citations": has_citations,
    }


def main():
    cases = json.loads(DATASET.read_text(encoding="utf-8"))

    results = []

    for case in cases:
        results.append(evaluate_case(case))

    total = len(results)
    retrieved = sum(r["source_found"] for r in results)
    cited = sum(r["has_citations"] for r in results)

    print("\n" + "=" * 55)
    print("EVALUATION SUMMARY")
    print(f"Cases evaluated: {total}")
    print(f"Expected source found: {retrieved}/{total}")
    print(f"Answers with citations: {cited}/{total}")
    print(
        "All citation IDs valid:",
        all(r["citation_ids_valid"] for r in results),
    )

    print(
        "\nNOTE: Factual correctness and citation support "
        "still require manual review."
    )


if __name__ == "__main__":
    main()
