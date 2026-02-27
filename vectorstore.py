"""Vector store operations for RAG.

This module provides functions to create and query a Chroma vector database
for document retrieval.

Run tests: uv run pytest tests/test_vectorstore.py
"""

from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Default configuration
CHROMA_DB_PATH = Path("./chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    """Get the embedding model instance. (Provided)"""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def create_vectorstore(
    chunks: list[str],
    metadatas: list[dict],
    collection_name: str = "papers",
    persist_directory: str | Path = CHROMA_DB_PATH,
) -> Chroma:
    """
    Create a Chroma vector store from document chunks.

    Args:
        chunks: List of text chunks to embed and store
        metadatas: List of metadata dicts (one per chunk), each with 'source' key
        collection_name: Name for the Chroma collection (default: "papers")
        persist_directory: Directory to persist the database (default: ./chroma_db)

    Returns:
        Chroma vector store instance

    Example:
        >>> chunks = ["RAG combines retrieval.", "LLMs can hallucinate."]
        >>> metadatas = [{"source": "paper1.pdf"}, {"source": "paper2.pdf"}]
        >>> vs = create_vectorstore(chunks, metadatas)

    TODO: Implement this function.
    - Use get_embeddings() to get the embedding model
    - Create Chroma vector store using Chroma.from_texts()
    - Persist to the specified directory
    """
    raise NotImplementedError("Implement create_vectorstore")


def load_vectorstore(
    collection_name: str = "papers",
    persist_directory: str | Path = CHROMA_DB_PATH,
) -> Chroma:
    """Load an existing Chroma vector store. (Provided)"""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )


def retrieve(vectorstore: Chroma, query: str, k: int = 3) -> list[Document]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Args:
        vectorstore: The Chroma vector store to search
        query: The search query
        k: Number of documents to retrieve (default: 3)

    Returns:
        List of Document objects with page_content and metadata

    TODO: Implement this function.
    - Use vectorstore.similarity_search()
    - Return top k results
    """
    raise NotImplementedError("Implement retrieve")


def retrieve_with_scores(
    vectorstore: Chroma, query: str, k: int = 3
) -> list[tuple[Document, float]]:
    """
    Retrieve top-k chunks with their similarity scores.

    Args:
        vectorstore: The Chroma vector store to search
        query: The search query
        k: Number of documents to retrieve (default: 3)

    Returns:
        List of (Document, score) tuples, sorted by relevance

    TODO: Implement this function.
    - Use vectorstore.similarity_search_with_score()
    - Return documents with their scores
    """
    raise NotImplementedError("Implement retrieve_with_scores")
