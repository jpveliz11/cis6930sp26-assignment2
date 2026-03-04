# Assignment 2: RAG System

Build a Retrieval-Augmented Generation (RAG) system that answers questions about research papers.

## Setup

```bash
# Install dependencies
uv sync

# Copy environment file and add your API key
cp .env.example .env
```

## Add Papers

Download at least 5 papers from the course reading list and place them in `papers/`:

```
papers/
├── lewis2020rag.pdf    # Required
├── wei2022cot.pdf      # Required
└── ...
```

## Your Task

Implement the TODO functions in:
- `chunker.py` - Document chunking (10 pts)
- `vectorstore.py` - Vector store operations (15 pts)
- `generator.py` - Answer generation (15 pts)
- `evaluate.py` - Evaluation metrics (10 pts)

## Running

```bash
# Run tests
uv run pytest

# Index papers
uv run python index.py

# Query the system
uv run python query.py "What is retrieval augmented generation?"

# Run evaluation
uv run python run_evaluation.py
```

## Evaluation Results

<!-- Fill in your results here -->

| Metric | Score |
|--------|-------|
| Precision@3 | 1.000 |
| Precision@5 | 1.000 |
| MRR | 1.000 |

### Test Queries Used

1. "What is retrieval augmented generation?"
2. "What's chain of thought?"
3. "Explain what Clinical Renal Replacement Therapy is ?"

## Chunking Strategy

<!-- Describe your chunking approach here -->

The system uses a sentence-based overlapping chunking strategy. Documents are first split into sentences using punctuation such as (., ?, !) then sentences are combined into chunks up to 500 characters. 

Each chunk includes a 50-character overlap from the previous chunk to preserve context and avoid losing information at chunk boundaries. If adding a sentence would exceed the chunk size a new chunk is started using the overlapping text. 

This approach keeps chunks small for efficient embeddings while maintaining enough context for accurate retrieval. 

## Example Queries

<!-- Show 2-3 example queries with answers -->

**Example 1: "What is retrieval augmented generation?"**

2026-03-04 15:59:23.183 | INFO     | __main__:main:28 - Retrieving relevant documents...
2026-03-04 15:59:23.299 | INFO     | __main__:main:35 - Found 3 relevant chunks
2026-03-04 15:59:23.299 | INFO     | __main__:main:38 - Generating answer...

============================================================
ANSWER:
============================================================
Retrieval‑augmented generation (RAG) is a framework that combines a pre‑trained parametric language model with an explicit non‑parametric memory accessed through a differentiable retrieval mechanism. In practice, a RAG model retrieves relevant documents or passages from an external index and conditions its generation on both the learned parameters and the retrieved content, enabling more factual and specific text generation. This approach is presented as a general‑purpose fine‑tuning recipe for language generation tasks【3】.

**Example 2: "What's chain of thought?"**

2026-03-04 16:13:54.813 | INFO     | __main__:main:28 - Retrieving relevant documents...
2026-03-04 16:13:54.923 | INFO     | __main__:main:35 - Found 3 relevant chunks
2026-03-04 16:13:54.923 | INFO     | __main__:main:38 - Generating answer...

============================================================
ANSWER:
============================================================
A **chain of thought** is a sequence of intermediate reasoning steps that a model generates to arrive at a final answer.  It mimics a step‑by‑step thought process, resembling a solution but is distinguished as a "chain of thought" to emphasize the sequential reasoning that leads to the answer rather than just the answer itself【3】.  This approach has been shown to be useful beyond merely activating knowledge, as the sequential reasoning embodied in the chain of thought can improve performance on certain tasks【1】【2】.

**Example 3: Explain what Clinical Renal Replacement Therapy is ?**

2026-03-04 16:18:03.179 | INFO     | __main__:main:28 - Retrieving relevant documents...
2026-03-04 16:18:03.276 | INFO     | __main__:main:35 - Found 3 relevant chunks
2026-03-04 16:18:03.276 | INFO     | __main__:main:38 - Generating answer...

============================================================
ANSWER:
============================================================
Clinical Renal Replacement Therapy (CRRT) is a continuous dialysis‑based treatment that is routinely used in Intensive Care Units (ICUs) to support patients who are suffering from Acute Kidney Injury (AKI) and other severe, multifactorial health conditions. It provides ongoing removal of waste products, fluid, and electrolytes, thereby helping to stabilize the patient's metabolic status while the kidneys recover or while the underlying disease is managed [3].

