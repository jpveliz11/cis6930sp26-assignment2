"""Tests for vectorstore module."""

import pytest
import tempfile
from pathlib import Path

from vectorstore import create_vectorstore, retrieve, retrieve_with_scores, load_vectorstore


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        "RAG combines retrieval with generation to reduce hallucination.",
        "Vector databases store embeddings for fast similarity search.",
        "Chain-of-thought prompting improves reasoning capabilities.",
    ]


@pytest.fixture
def sample_metadatas():
    """Sample metadata for testing."""
    return [
        {"source": "rag_paper.pdf", "chunk_id": 0},
        {"source": "vector_db.pdf", "chunk_id": 0},
        {"source": "cot_paper.pdf", "chunk_id": 0},
    ]


class TestCreateVectorstore:
    def test_creates_vectorstore(self, sample_chunks, sample_metadatas):
        """Test vector store creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vs = create_vectorstore(
                sample_chunks,
                sample_metadatas,
                persist_directory=tmpdir
            )
            assert vs is not None

    def test_stores_all_chunks(self, sample_chunks, sample_metadatas):
        """Test that all chunks are stored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vs = create_vectorstore(
                sample_chunks,
                sample_metadatas,
                persist_directory=tmpdir
            )
            # Chroma stores the count in _collection
            assert vs._collection.count() == len(sample_chunks)


class TestRetrieve:
    def test_retrieve_returns_documents(self, sample_chunks, sample_metadatas):
        """Test retrieval returns documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vs = create_vectorstore(
                sample_chunks,
                sample_metadatas,
                persist_directory=tmpdir
            )
            docs = retrieve(vs, "What is RAG?", k=2)

            assert len(docs) <= 2
            assert all(hasattr(d, "page_content") for d in docs)
            assert all(hasattr(d, "metadata") for d in docs)

    def test_retrieve_relevant_content(self, sample_chunks, sample_metadatas):
        """Test retrieval finds relevant content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vs = create_vectorstore(
                sample_chunks,
                sample_metadatas,
                persist_directory=tmpdir
            )
            docs = retrieve(vs, "retrieval augmented generation", k=1)

            assert len(docs) == 1
            assert "RAG" in docs[0].page_content or "retrieval" in docs[0].page_content.lower()


class TestRetrieveWithScores:
    def test_returns_scores(self, sample_chunks, sample_metadatas):
        """Test retrieval returns scores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vs = create_vectorstore(
                sample_chunks,
                sample_metadatas,
                persist_directory=tmpdir
            )
            results = retrieve_with_scores(vs, "What is RAG?", k=2)

            assert len(results) <= 2
            assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
            assert all(isinstance(r[1], float) for r in results)
