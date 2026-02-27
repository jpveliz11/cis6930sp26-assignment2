"""Query the RAG system.

Usage: uv run python query.py "What is retrieval augmented generation?"
"""

import sys

from loguru import logger

from vectorstore import load_vectorstore, retrieve
from generator import generate_answer_with_citations


def main():
    """Query the RAG system."""
    if len(sys.argv) < 2:
        print("Usage: uv run python query.py \"Your question here\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    logger.info(f"Query: {query}")

    # Load vector store
    logger.info("Loading vector store...")
    vectorstore = load_vectorstore()

    # Retrieve relevant documents
    logger.info("Retrieving relevant documents...")
    docs = retrieve(vectorstore, query, k=3)

    if not docs:
        print("No relevant documents found.")
        return

    logger.info(f"Found {len(docs)} relevant chunks")

    # Generate answer with citations
    logger.info("Generating answer...")
    result = generate_answer_with_citations(query, docs)

    # Print results
    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(result["answer"])

    print("\n" + "-" * 60)
    print("SOURCES:")
    print("-" * 60)
    for citation in result["citations"]:
        print(f"  - {citation.get('source', 'Unknown')}")

    print()


if __name__ == "__main__":
    main()
