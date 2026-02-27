"""Evaluation metrics for RAG retrieval.

This module provides functions to evaluate retrieval quality using
standard information retrieval metrics.

Run tests: uv run pytest tests/test_evaluate.py
"""


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Calculate precision@k for retrieval evaluation.

    Precision@k = (# relevant docs in top k) / k

    Args:
        retrieved_ids: List of retrieved document IDs in ranked order
        relevant_ids: List of document IDs that are actually relevant
        k: Number of top results to consider

    Returns:
        Precision@k score between 0.0 and 1.0

    Example:
        >>> retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        >>> relevant = ["doc1", "doc3", "doc7"]
        >>> precision_at_k(retrieved, relevant, k=3)
        0.6666666666666666

    TODO: Implement this function.
    """
    raise NotImplementedError("Implement precision_at_k")


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Calculate recall@k for retrieval evaluation. (Provided)

    Recall@k = (# relevant docs in top k) / (# total relevant docs)
    """
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(top_k & relevant_set) / len(relevant_set)


def mean_reciprocal_rank(queries_results: list[tuple[list[str], str]]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) across multiple queries.

    MRR = (1/n) * sum(1/rank_i) where rank_i is position of first relevant doc

    Args:
        queries_results: List of (retrieved_ids, first_relevant_id) tuples

    Returns:
        MRR score between 0.0 and 1.0

    Example:
        >>> results = [
        ...     (["doc1", "doc2", "doc3"], "doc1"),  # Rank 1 -> RR = 1.0
        ...     (["doc4", "doc5", "doc6"], "doc5"),  # Rank 2 -> RR = 0.5
        ...     (["doc7", "doc8", "doc9"], "doc10"), # Not found -> RR = 0.0
        ... ]
        >>> mean_reciprocal_rank(results)
        0.5

    TODO: Implement this function.
    """
    raise NotImplementedError("Implement mean_reciprocal_rank")
