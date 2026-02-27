"""Answer generation for RAG.

This module provides functions to generate answers based on retrieved context,
with support for source citations.

Run tests: uv run pytest tests/test_generator.py
"""

import re

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.1) -> ChatOpenAI:
    """Get a configured LLM instance. (Provided)"""
    return ChatOpenAI(model=model, temperature=temperature)


def format_context(docs: list[Document]) -> str:
    """Format retrieved documents into a numbered context string. (Provided)"""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        page_str = f", page {page}" if page else ""
        parts.append(f"[{i}] Source: {source}{page_str}\n{doc.page_content}")
    return "\n\n".join(parts)


def generate_answer(query: str, context_docs: list[Document], llm=None) -> str:
    """
    Generate an answer based on retrieved context.

    Args:
        query: The user's question
        context_docs: Retrieved documents to use as context
        llm: The language model (if None, creates default with get_llm())

    Returns:
        Generated answer string

    TODO: Implement this function.
    - If llm is None, use get_llm()
    - Format context using format_context()
    - Create a prompt asking the model to answer based on the context
    - Call llm.invoke(prompt) and return .content
    """
    raise NotImplementedError("Implement generate_answer")


def generate_answer_with_citations(
    query: str, context_docs: list[Document], llm=None
) -> dict:
    """
    Generate an answer with explicit citations to source documents.

    Args:
        query: The user's question
        context_docs: Retrieved documents to use as context
        llm: The language model (if None, creates default)

    Returns:
        Dictionary with:
        - "answer": The generated answer text
        - "citations": List of cited source metadata dicts

    TODO: Implement this function.
    - Create a prompt instructing the model to cite sources as [1], [2], etc.
    - Parse citations from the response using _extract_citations()
    - Return structured output
    """
    raise NotImplementedError("Implement generate_answer_with_citations")


def _extract_citations(text: str, max_source: int) -> list[int]:
    """Extract citation numbers [1], [2], etc. from text. (Provided)"""
    pattern = r"\[(\d+)\]"
    matches = re.findall(pattern, text)
    citations = []
    for match in matches:
        num = int(match)
        if 1 <= num <= max_source and num not in citations:
            citations.append(num)
    return citations
