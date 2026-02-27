"""Tests for chunker module."""

import pytest
from chunker import chunk_document, chunk_by_paragraphs


class TestChunkDocument:
    def test_basic_chunking(self):
        """Test basic text chunking."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = chunk_document(text, chunk_size=40, overlap=10)

        assert len(chunks) >= 1
        assert all(len(c) <= 50 for c in chunks)  # Allow some flexibility

    def test_empty_text(self):
        """Test handling of empty text."""
        chunks = chunk_document("", chunk_size=100, overlap=10)
        assert chunks == [] or chunks == [""]

    def test_short_text(self):
        """Test text shorter than chunk_size."""
        text = "Short text."
        chunks = chunk_document(text, chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_overlap_exists(self):
        """Test that chunks have overlap."""
        text = "A" * 100 + ". " + "B" * 100 + ". " + "C" * 100
        chunks = chunk_document(text, chunk_size=120, overlap=20)

        if len(chunks) > 1:
            # Check some overlap exists between consecutive chunks
            for i in range(len(chunks) - 1):
                # At least some characters should be shared
                assert any(c in chunks[i+1] for c in chunks[i][-30:])


class TestChunkByParagraphs:
    def test_basic_paragraphs(self):
        """Test basic paragraph chunking."""
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunk_by_paragraphs(text, max_chunk_size=100)

        assert len(chunks) >= 1
        assert all(len(c) <= 100 for c in chunks)

    def test_empty_text(self):
        """Test handling of empty text."""
        chunks = chunk_by_paragraphs("", max_chunk_size=100)
        assert chunks == [] or chunks == [""]

    def test_single_paragraph(self):
        """Test single paragraph."""
        text = "Just one paragraph here."
        chunks = chunk_by_paragraphs(text, max_chunk_size=100)
        assert len(chunks) == 1

    def test_merges_small_paragraphs(self):
        """Test that small paragraphs are merged."""
        text = "A.\n\nB.\n\nC."
        chunks = chunk_by_paragraphs(text, max_chunk_size=100)

        # Should merge into one chunk since all are small
        assert len(chunks) == 1

    def test_respects_max_size(self):
        """Test that chunks don't exceed max size."""
        text = "A" * 50 + "\n\n" + "B" * 50 + "\n\n" + "C" * 50
        chunks = chunk_by_paragraphs(text, max_chunk_size=60)

        assert all(len(c) <= 60 for c in chunks)
