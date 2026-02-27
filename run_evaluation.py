"""Run evaluation on the RAG system.

Usage: uv run python run_evaluation.py
"""

from loguru import logger

from vectorstore import load_vectorstore, retrieve
from evaluate import precision_at_k, mean_reciprocal_rank


# Define test queries with known relevant documents
# Format: (query, [list of relevant source filenames])
TEST_QUERIES = [
    (
        "What is retrieval augmented generation?",
        ["lewis2020rag.pdf"],
    ),
    (
        "How does chain of thought prompting work?",
        ["wei2022cot.pdf"],
    ),
    # Add more test queries here
]


def evaluate_retrieval(vectorstore, queries: list[tuple[str, list[str]]], k: int = 5):
    """Evaluate retrieval performance."""

    precision_scores = []
    mrr_data = []

    for query, relevant_sources in queries:
        logger.info(f"Query: {query[:50]}...")

        # Retrieve documents
        docs = retrieve(vectorstore, query, k=k)
        retrieved_sources = [doc.metadata.get("source", "") for doc in docs]

        # Calculate precision@k
        p_at_k = precision_at_k(retrieved_sources, relevant_sources, k)
        precision_scores.append(p_at_k)

        # Prepare data for MRR (use first relevant source)
        if relevant_sources:
            mrr_data.append((retrieved_sources, relevant_sources[0]))

        logger.info(f"  Retrieved: {retrieved_sources[:3]}")
        logger.info(f"  Precision@{k}: {p_at_k:.3f}")

    # Calculate MRR
    mrr = mean_reciprocal_rank(mrr_data) if mrr_data else 0.0

    return {
        "precision_at_k": sum(precision_scores) / len(precision_scores) if precision_scores else 0.0,
        "mrr": mrr,
        "k": k,
        "num_queries": len(queries),
    }


def main():
    """Run evaluation."""
    logger.info("Loading vector store...")
    vectorstore = load_vectorstore()

    logger.info(f"Running evaluation on {len(TEST_QUERIES)} queries...")

    # Evaluate at different k values
    for k in [3, 5]:
        results = evaluate_retrieval(vectorstore, TEST_QUERIES, k=k)

        print(f"\n{'=' * 40}")
        print(f"Evaluation Results (k={k})")
        print(f"{'=' * 40}")
        print(f"Number of queries: {results['num_queries']}")
        print(f"Precision@{k}: {results['precision_at_k']:.3f}")
        print(f"MRR: {results['mrr']:.3f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
