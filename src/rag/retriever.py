"""Reusable retrieval utilities for local ChromaDB vector stores.

This module loads a caller-provided ChromaDB store and exposes a small
retrieval interface that can be reused by multiple agents. It does not call
any LLMs or generate answers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
NO_RESULTS_MESSAGE = "No relevant information found."


def _resolve_persist_directory(persist_directory: Path) -> Path:
    """Resolve a vector store path relative to the project root when needed.

    Args:
        persist_directory: Absolute path or project-relative path to ChromaDB.

    Returns:
        Resolved absolute path to the ChromaDB directory.
    """

    if persist_directory.is_absolute():
        return persist_directory

    return PROJECT_ROOT / persist_directory


def load_vector_store(persist_directory: Path = VECTOR_DB_DIR) -> Any:
    """Load an existing ChromaDB vector store.

    Args:
        persist_directory: Directory containing the persisted ChromaDB files.

    Returns:
        A LangChain Chroma vector store instance.

    Raises:
        FileNotFoundError: If the vector database directory does not exist.
        RuntimeError: If the vector store cannot be loaded.
    """

    resolved_directory = _resolve_persist_directory(persist_directory)

    if not resolved_directory.exists():
        raise FileNotFoundError(
            f"Vector database not found at {resolved_directory}. "
            "Create embeddings for this document domain first."
        )

    try:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        return Chroma(
            persist_directory=str(resolved_directory),
            embedding_function=embeddings,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load vector store from {resolved_directory}") from exc


def get_retriever(
    persist_directory: Path = VECTOR_DB_DIR,
    top_k: int = DEFAULT_TOP_K,
) -> Any:
    """Create a reusable LangChain retriever for a vector store.

    Args:
        persist_directory: Directory containing the persisted ChromaDB files.
        top_k: Number of relevant chunks to retrieve.

    Returns:
        A LangChain retriever configured for MMR search.

    Raises:
        ValueError: If ``top_k`` is less than 1.
    """

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    vector_store = load_vector_store(persist_directory)
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": top_k,
            "fetch_k": 15,
            "lambda_mult": 0.7,
        },
    )


def retrieve_context(
    query: str,
    persist_directory: Path = VECTOR_DB_DIR,
    top_k: int = DEFAULT_TOP_K,
) -> str:
    """Retrieve relevant chunks from a caller-provided ChromaDB store.

    Args:
        query: User query to search against the vector store.
        persist_directory: Directory containing the persisted ChromaDB files.
        top_k: Number of relevant chunks to retrieve.

    Returns:
        A formatted string containing the retrieved chunks, or
        ``NO_RESULTS_MESSAGE`` when no relevant chunks are found.

    Raises:
        ValueError: If the query is empty or ``top_k`` is invalid.
        RuntimeError: If retrieval fails.
    """

    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("query must not be empty.")

    try:
        retriever = get_retriever(
            persist_directory=persist_directory,
            top_k=top_k,
        )
        documents = retriever.invoke(cleaned_query)
    except Exception as exc:
        raise RuntimeError(f"Failed to retrieve context from vector store: {exc}") from exc

    if not documents:
        return NO_RESULTS_MESSAGE

    formatted_chunks: list[str] = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown source")
        content = document.page_content.strip()
        if content:
            formatted_chunks.append(f"[Chunk {index} | Source: {source}]\n{content}")

    if not formatted_chunks:
        return NO_RESULTS_MESSAGE

    return "\n\n---\n\n".join(formatted_chunks)


def main() -> None:
    """Demonstrate retrieval from HR and Technical vector stores."""

    examples = [
        ("HR", "How many leaves are allowed?", Path("vector_db/hr")),
        ("Technical", "How do I install Docker?", Path("vector_db/technical")),
    ]

    for label, query, persist_directory in examples:
        print(f"{label} Query: {query}")
        print(f"Vector Store: {persist_directory}")
        print()

        try:
            print(retrieve_context(query, persist_directory))
        except Exception as exc:
            print(f"{label} retrieval failed: {exc}", file=sys.stderr)

        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
