"""Tests for evaluate module."""

import pytest
from evaluate import precision_at_k, recall_at_k, mean_reciprocal_rank


class TestPrecisionAtK:
    def test_all_relevant(self):
        """Test when all retrieved are relevant."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc2", "doc3"]

        assert precision_at_k(retrieved, relevant, k=3) == 1.0

    def test_none_relevant(self):
        """Test when none retrieved are relevant."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc4", "doc5"]

        assert precision_at_k(retrieved, relevant, k=3) == 0.0

    def test_partial_relevant(self):
        """Test partial relevance."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc1", "doc3", "doc7"]

        # 2 relevant in top 3: precision = 2/3
        assert abs(precision_at_k(retrieved, relevant, k=3) - 2/3) < 0.001

    def test_k_larger_than_retrieved(self):
        """Test when k is larger than retrieved list."""
        retrieved = ["doc1", "doc2"]
        relevant = ["doc1"]

        # Should handle gracefully
        result = precision_at_k(retrieved, relevant, k=5)
        assert 0.0 <= result <= 1.0

    def test_empty_retrieved(self):
        """Test empty retrieved list."""
        assert precision_at_k([], ["doc1"], k=3) == 0.0


class TestRecallAtK:
    def test_all_found(self):
        """Test when all relevant are found."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc2"]

        assert recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_none_found(self):
        """Test when none relevant are found."""
        retrieved = ["doc4", "doc5", "doc6"]
        relevant = ["doc1", "doc2"]

        assert recall_at_k(retrieved, relevant, k=3) == 0.0


class TestMeanReciprocalRank:
    def test_first_position(self):
        """Test relevant doc in first position."""
        results = [(["doc1", "doc2", "doc3"], "doc1")]
        assert mean_reciprocal_rank(results) == 1.0

    def test_second_position(self):
        """Test relevant doc in second position."""
        results = [(["doc1", "doc2", "doc3"], "doc2")]
        assert mean_reciprocal_rank(results) == 0.5

    def test_not_found(self):
        """Test relevant doc not found."""
        results = [(["doc1", "doc2", "doc3"], "doc4")]
        assert mean_reciprocal_rank(results) == 0.0

    def test_multiple_queries(self):
        """Test MRR across multiple queries."""
        results = [
            (["doc1", "doc2", "doc3"], "doc1"),  # RR = 1.0
            (["doc4", "doc5", "doc6"], "doc5"),  # RR = 0.5
            (["doc7", "doc8", "doc9"], "doc10"), # RR = 0.0
        ]
        # MRR = (1.0 + 0.5 + 0.0) / 3 = 0.5
        assert mean_reciprocal_rank(results) == 0.5

    def test_empty_input(self):
        """Test empty input."""
        assert mean_reciprocal_rank([]) == 0.0
