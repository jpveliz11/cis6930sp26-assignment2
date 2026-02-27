"""Tests for generator module."""

import pytest
from unittest.mock import Mock, patch
from langchain.schema import Document

from generator import format_context, generate_answer, generate_answer_with_citations, _extract_citations


@pytest.fixture
def sample_docs():
    """Sample documents for testing."""
    return [
        Document(
            page_content="RAG combines retrieval with generation.",
            metadata={"source": "lewis2020.pdf", "page": 1}
        ),
        Document(
            page_content="This reduces hallucination in LLMs.",
            metadata={"source": "lewis2020.pdf", "page": 2}
        ),
    ]


class TestFormatContext:
    def test_formats_documents(self, sample_docs):
        """Test context formatting."""
        context = format_context(sample_docs)

        assert "[1]" in context
        assert "[2]" in context
        assert "lewis2020.pdf" in context
        assert "RAG combines" in context


class TestExtractCitations:
    def test_extracts_citations(self):
        """Test citation extraction."""
        text = "According to [1], RAG works well. See also [2] and [1]."
        citations = _extract_citations(text, max_source=3)

        assert 1 in citations
        assert 2 in citations
        assert len(citations) == 2  # No duplicates

    def test_ignores_invalid_citations(self):
        """Test that invalid citations are ignored."""
        text = "See [5] and [10]."
        citations = _extract_citations(text, max_source=3)

        assert citations == []


class TestGenerateAnswer:
    @patch("generator.get_llm")
    def test_generates_answer(self, mock_get_llm, sample_docs):
        """Test answer generation."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="RAG reduces hallucination.")
        mock_get_llm.return_value = mock_llm

        answer = generate_answer("What does RAG do?", sample_docs)

        assert isinstance(answer, str)
        assert len(answer) > 0

    @patch("generator.get_llm")
    def test_uses_provided_llm(self, mock_get_llm, sample_docs):
        """Test that provided LLM is used."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="Test answer")

        answer = generate_answer("Test?", sample_docs, llm=mock_llm)

        mock_get_llm.assert_not_called()
        mock_llm.invoke.assert_called_once()


class TestGenerateAnswerWithCitations:
    @patch("generator.get_llm")
    def test_returns_structured_output(self, mock_get_llm, sample_docs):
        """Test structured output with citations."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(
            content="RAG reduces hallucination [1]. It combines retrieval [2]."
        )
        mock_get_llm.return_value = mock_llm

        result = generate_answer_with_citations("What is RAG?", sample_docs)

        assert "answer" in result
        assert "citations" in result
        assert isinstance(result["citations"], list)
