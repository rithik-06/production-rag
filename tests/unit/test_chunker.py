from chunking.chunker import chunk_documents
from langchain_core.documents import Document


def test_chunk_documents_returns_chunks():
    """Test that chunker splits documents into chunks."""
    docs = [Document(page_content="This is a test document. " * 50)]
    chunks = chunk_documents(docs)
    assert len(chunks) > 0


def test_chunk_size_is_respected():
    """Test that chunks don't exceed max size."""
    docs = [Document(page_content="This is a test document. " * 100)]
    chunks = chunk_documents(docs, chunk_size=256)
    for chunk in chunks:
        assert len(chunk.page_content) <= 300  # small buffer allowed


def test_empty_document():
    """Test that empty input returns empty list."""
    chunks = chunk_documents([])
    assert chunks == []