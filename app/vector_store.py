
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.document_processing import (
    load_documents,
    split_documents,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION = "shopease_policies"

load_dotenv(PROJECT_ROOT / ".env")


def get_embeddings():
    """Initialize the embedding model."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing")

    return OpenAIEmbeddings(
        model="text-embedding-3-small"
    )


def build_vector_store():
    """Index ShopEase policy chunks in Chroma."""

    documents = load_documents()
    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("No document chunks to index")

    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(DB_PATH),
    )

    # Rebuild the small V1 collection without duplicates.
    existing = store.get()
    existing_ids = existing["ids"]

    if existing_ids:
        store.delete(ids=existing_ids)

    ids = [
        chunk.metadata["chunk_id"]
        for chunk in chunks
    ]

    store.add_documents(
        documents=chunks,
        ids=ids,
    )

    print(f"Indexed {len(chunks)} chunks")
    print(f"Stored in: {DB_PATH}")
    print(f"Collection size: {store._collection.count()}")

    return store


def search_documents(query: str, k: int = 3):
    """Search the existing knowledge base."""

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if not DB_PATH.exists():
        raise FileNotFoundError(
            "Vector database missing. Build it first."
        )

    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(DB_PATH),
        create_collection_if_not_exists=False,
    )

    return store.similarity_search_with_score(
        query=query,
        k=k,
    )


if __name__ == "__main__":
    build_vector_store()
