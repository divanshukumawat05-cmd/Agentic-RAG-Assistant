"""RAG-based HR agent backed by local HR documents.

The agent retrieves HR context from ChromaDB before calling Gemini. It never
asks Gemini to answer from the model's own knowledge.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
NO_RELEVANT_INFO_MESSAGE = "No relevant HR information found."

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag.retriever import retrieve_context  # noqa: E402


def _load_llm() -> ChatGoogleGenerativeAI:
    """Create a Gemini chat model using the API key from the project .env file.

    Returns:
        Configured ``ChatGoogleGenerativeAI`` instance.

    Raises:
        RuntimeError: If ``GOOGLE_API_KEY`` is missing.
    """

    load_dotenv(PROJECT_ROOT / ".env")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not configured in the .env file.")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key,
        temperature=0,
    )


def _build_prompt(context: str, user_query: str) -> str:
    """Build the strict HR RAG prompt for Gemini.

    Args:
        context: Retrieved HR document context.
        user_query: User's HR question.

    Returns:
        Prompt string constrained to the provided HR context.
    """

    return f"""
You are the HR assistant of the company.

Answer ONLY using the provided HR context.

If the answer cannot be found in the context,
reply exactly:

"I could not find this information in the HR documents."

Never use your own knowledge.

HR Context:
{context}

User Question:
{user_query}
"""


def hr_agent(user_query: str) -> str:
    """Answer an HR question using retrieved HR document context only.

    Args:
        user_query: User's HR question.

    Returns:
        Gemini's answer grounded in retrieved HR context, or the exact no-results
        message when the retriever finds no relevant HR information.

    Raises:
        ValueError: If ``user_query`` is empty.
        RuntimeError: If retrieval or generation fails.
    """

    cleaned_query = user_query.strip()
    if not cleaned_query:
        raise ValueError("user_query must not be empty.")

    try:
        context = retrieve_context(
            cleaned_query,
            Path("vector_db/hr"),
        )
    except Exception as exc:
        raise RuntimeError("Failed to retrieve HR context.") from exc

    if context == NO_RELEVANT_INFO_MESSAGE:
        return NO_RELEVANT_INFO_MESSAGE

    prompt = _build_prompt(context=context, user_query=cleaned_query)

    try:
        llm = _load_llm()
        response = llm.invoke(prompt)
    except Exception as exc:
        raise RuntimeError("Failed to generate HR answer with Gemini.") from exc

    answer = getattr(response, "content", "")
    if isinstance(answer, str):
        return answer.strip()

    return str(answer).strip()


def main() -> None:
    """Run a simple HR agent smoke test."""

    test_query = "How many leaves are allowed during internship?"

    try:
        print(hr_agent(test_query))
    except Exception as exc:
        print(f"HR agent failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
