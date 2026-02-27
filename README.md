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
| Precision@3 | |
| Precision@5 | |
| MRR | |

### Test Queries Used

1.
2.
3.

## Chunking Strategy

<!-- Describe your chunking approach here -->

## Example Queries

<!-- Show 2-3 example queries with answers -->
