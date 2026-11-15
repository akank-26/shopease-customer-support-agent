
from app.vector_store import search_documents


def main():
    query = "Can I send back my earphones?"

    print("Question:", query)

    results = search_documents(
        query=query,
        k=3,
    )

    for rank, (document, distance) in enumerate(
        results, start=1
    ):
        print(f"\n--- Result {rank} ---")
        print("Source:", document.metadata["source"])
        print("Chunk ID:", document.metadata["chunk_id"])
        print("Distance:", round(distance, 4))
        print("Content:")
        print(document.page_content)


if __name__ == "__main__":
    main()
