"""Build local ChromaDB vector stores for the Agentic RAG project.

This script loads Markdown files from configured document directories, splits
them with LangChain's recursive splitter, embeds them with HuggingFace sentence
transformers, and persists each collection into its own ChromaDB directory.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HR_DOCS_DIR = PROJECT_ROOT / "data" / "hr_docs"
TECHNICAL_DOCS_DIR = PROJECT_ROOT / "data" / "technical_docs"
HR_VECTOR_DB_DIR = PROJECT_ROOT / "vector_db" / "hr"
TECHNICAL_VECTOR_DB_DIR = PROJECT_ROOT / "vector_db" / "technical"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents(docs_dir: Path) -> list[Document]:
    """Load all Markdown documents recursively from a document directory.

    Args:
        docs_dir: Directory containing Markdown documents.

    Returns:
        A list of LangChain ``Document`` objects.

    Raises:
        FileNotFoundError: If the documents directory does not exist.
        ValueError: If no Markdown files are found.
        RuntimeError: If a Markdown file cannot be read.
    """

    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    markdown_files = sorted(docs_dir.rglob("*.md"))
    if not markdown_files:
        raise ValueError(f"No Markdown files found in: {docs_dir}")

    documents: list[Document] = []
    for markdown_file in markdown_files:
        try:
            content = markdown_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"Failed to read document: {markdown_file}") from exc

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": str(markdown_file.relative_to(PROJECT_ROOT)),
                    "file_name": markdown_file.name,
                },
            )
        )

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into chunks with LangChain's recursive text splitter.

    Args:
        documents: Documents to split.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        Chunked LangChain documents.

    Raises:
        ValueError: If no chunks are produced.
    """

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Document splitting produced no chunks.")

    return chunks


def create_vector_store(docs_directory: Path, persist_directory: Path) -> Any:
    """Create a fresh ChromaDB vector store from Markdown documents.

    Existing vector store data is removed before the new store is created.

    Args:
        docs_directory: Directory containing Markdown documents to embed.
        persist_directory: Local directory where ChromaDB data is stored.

    Returns:
        The created LangChain Chroma vector store instance.

    Raises:
        RuntimeError: If the existing vector database cannot be removed.
    """

    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    documents = load_documents(docs_directory)
    chunks = split_documents(documents)

    if persist_directory.exists():
        try:
            shutil.rmtree(persist_directory)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to remove existing vector database: {persist_directory}"
            ) from exc

    persist_directory.mkdir(parents=True, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_directory),
    )

    persist = getattr(vector_store, "persist", None)
    if callable(persist):
        persist()

    return vector_store


def main() -> None:
    """Create HR and Technical ChromaDB vector stores."""

    try:
        print("Creating HR Vector Store...")
        create_vector_store(
            docs_directory=HR_DOCS_DIR,
            persist_directory=HR_VECTOR_DB_DIR,
        )
        print("Completed.")

        print("Creating Technical Vector Store...")
        create_vector_store(
            docs_directory=TECHNICAL_DOCS_DIR,
            persist_directory=TECHNICAL_VECTOR_DB_DIR,
        )
        print("Completed.")
    except Exception as exc:
        print(f"Embedding pipeline failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
