
import pytest

from langchain_core.documents import Document

from app.document_processing import (
    load_documents,
    split_documents,
)


def test_loads_four_documents():
    documents = load_documents()

    assert len(documents) == 4


def test_documents_have_source_metadata():
    documents = load_documents()

    for doc in documents:
        assert doc.metadata["source"]
        assert doc.metadata["document_id"]
        assert doc.metadata["version"]


def test_chunks_preserve_metadata():
    documents = load_documents()
    chunks = split_documents(documents)

    assert len(chunks) >= len(documents)

    for chunk in chunks:
        assert chunk.metadata["source"]
        assert chunk.metadata["document_id"]
        assert chunk.metadata["chunk_id"]
        assert len(chunk.page_content) <= 500


def test_missing_document_raises_error(
    monkeypatch, tmp_path
):
    import app.document_processing as processing

    monkeypatch.setattr(
        processing,
        "KNOWLEDGE_BASE",
        tmp_path,
    )

    with pytest.raises(FileNotFoundError):
        processing.load_documents()


def test_empty_document_list():
    chunks = split_documents([])

    assert chunks == []


def test_chunk_ids_are_unique():
    chunks = split_documents(load_documents())

    ids = [chunk.metadata["chunk_id"] for chunk in chunks]

    assert len(ids) == len(set(ids))
