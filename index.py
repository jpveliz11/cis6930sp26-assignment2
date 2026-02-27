"""Index papers into the vector store.

Usage: uv run python index.py
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from loguru import logger

from chunker import chunk_document
from vectorstore import create_vectorstore


PAPERS_DIR = Path("papers")


def load_papers(papers_dir: Path = PAPERS_DIR) -> list[tuple[str, str]]:
    """Load all PDFs from the papers directory.

    Returns:
        List of (filename, text) tuples
    """
    papers = []
    pdf_files = list(papers_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {papers_dir}")
        return papers

    for pdf_path in pdf_files:
        logger.info(f"Loading {pdf_path.name}")
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            text = "\n\n".join(page.page_content for page in pages)
            papers.append((pdf_path.name, text))
        except Exception as e:
            logger.error(f"Failed to load {pdf_path.name}: {e}")

    return papers


def main():
    """Index all papers into the vector store."""
    logger.info("Starting indexing...")

    # Load papers
    papers = load_papers()
    if not papers:
        logger.error("No papers to index. Add PDFs to the papers/ directory.")
        return

    logger.info(f"Loaded {len(papers)} papers")

    # Chunk all papers
    all_chunks = []
    all_metadatas = []

    for filename, text in papers:
        chunks = chunk_document(text, chunk_size=500, overlap=50)
        logger.info(f"  {filename}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                "source": filename,
                "chunk_id": i,
            })

    logger.info(f"Total chunks: {len(all_chunks)}")

    # Create vector store
    logger.info("Creating vector store...")
    vectorstore = create_vectorstore(all_chunks, all_metadatas)

    logger.info("Indexing complete!")
    logger.info(f"Vector store saved to ./chroma_db")


if __name__ == "__main__":
    main()
