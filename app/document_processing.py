
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE = PROJECT_ROOT / "data" / "knowledge_base"

EXPECTED_FILES = [
    "return_policy.md",
    "refund_policy.md",
    "shipping_policy.md",
    "contact_support.md",
]


def load_documents() -> list[Document]:
    """Load ShopEase policies as LangChain Documents."""
    documents = []

    for filename in EXPECTED_FILES:
        path = KNOWLEDGE_BASE / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Knowledge base file missing: {path}"
            )

        content = path.read_text(encoding="utf-8")

        def get_field(field: str) -> str:
            prefix = f"{field}:"

            for line in content.splitlines():
                if line.startswith(prefix):
                    return line.split(":", 1)[1].strip()

            raise ValueError(
                f"Missing {field} in {filename}"
            )

        document = Document(
            page_content=content,
            metadata={
                "source": filename,
                "document_id": get_field("Document ID"),
                "version": get_field("Version"),
            },
        )

        documents.append(document)

    return documents


def split_documents(
    documents: list[Document],
) -> list[Document]:
    """Split documents and preserve source metadata."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    counters = {}

    for chunk in chunks:
        document_id = chunk.metadata["document_id"]

        counters[document_id] = counters.get(document_id, 0) + 1

        chunk.metadata["chunk_id"] = (
            f"{document_id}-{counters[document_id]:03d}"
        )

    return chunks


if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)

    print(f"Loaded documents: {len(documents)}")
    print(f"Generated chunks: {len(chunks)}")

    for chunk in chunks:
        print("\n------------------------------")
        print("Chunk ID:", chunk.metadata["chunk_id"])
        print("Source:", chunk.metadata["source"])
        print("Characters:", len(chunk.page_content))
        print("Content:")
        print(chunk.page_content)
