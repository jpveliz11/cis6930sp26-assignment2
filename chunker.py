"""Document chunking strategies for RAG.

This module provides functions to split documents into smaller chunks
suitable for embedding and retrieval.

Run tests: uv run pytest tests/test_chunker.py
"""


def chunk_document(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split a document into overlapping chunks.

    This function splits text into chunks of approximately `chunk_size` characters,
    with `overlap` characters shared between consecutive chunks. It attempts to
    split on sentence boundaries when possible.

    Args:
        text: The document text to chunk
        chunk_size: Maximum characters per chunk (default: 500)
        overlap: Number of characters to overlap between chunks (default: 50)

    Returns:
        List of text chunks

    Example:
        >>> text = "First sentence. Second sentence. Third sentence."
        >>> chunks = chunk_document(text, chunk_size=30, overlap=10)
        >>> len(chunks) > 1
        True

    TODO: Implement this function.
    - Split on sentence boundaries when possible (look for ". ", "? ", "! ")
    - Ensure chunks don't exceed chunk_size
    - Include overlap characters from previous chunk
    - Handle edge cases (empty text, very long sentences)
    """
    raise NotImplementedError("Implement chunk_document")


def chunk_by_paragraphs(text: str, max_chunk_size: int = 1000) -> list[str]:
    """
    Split a document by paragraphs, merging small paragraphs.

    This function splits text on paragraph boundaries (double newlines) and
    merges consecutive small paragraphs to create chunks closer to max_chunk_size.
    It never splits a paragraph in the middle.

    Args:
        text: The document text to chunk
        max_chunk_size: Maximum characters per chunk (default: 1000)

    Returns:
        List of text chunks, each containing one or more complete paragraphs

    Example:
        >>> text = "Para 1.\\n\\nPara 2.\\n\\nPara 3."
        >>> chunks = chunk_by_paragraphs(text, max_chunk_size=50)
        >>> len(chunks) >= 1
        True

    TODO: Implement this function.
    - Split on double newlines ("\\n\\n") to get paragraphs
    - Merge consecutive small paragraphs until adding another would exceed max_chunk_size
    - Don't split paragraphs mid-text
    - Handle edge cases (single paragraph, empty paragraphs)
    """
    raise NotImplementedError("Implement chunk_by_paragraphs")
