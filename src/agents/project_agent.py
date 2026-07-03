"""Project Management agent backed by the interns SQLite database.

The agent selects a reusable SQL query helper, converts the returned DataFrame
to text, and asks Gemini to answer only from those SQL results.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from database.sql_queries import (  # noqa: E402
    find_by_mentor,
    find_by_project,
    find_by_status,
    find_intern_by_name,
    show_all_interns,
)


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


def _dataframe_to_text(dataframe: pd.DataFrame) -> str:
    """Convert SQL query results into prompt-ready text.

    Args:
        dataframe: SQL results returned by a query helper.

    Returns:
        Text representation of the DataFrame, or a no-data marker.
    """

    if dataframe.empty:
        return "No data available."

    return dataframe.to_string(index=False)


def _build_prompt(results: str, user_query: str) -> str:
    """Build the strict SQL-results-only Project Agent prompt.

    Args:
        results: Text representation of SQL query results.
        user_query: User's project-related question.

    Returns:
        Prompt string constrained to the provided SQL results.
    """

    return f"""You are the Project Management Assistant.

Answer ONLY using the SQL results provided below.

If no data is available, say:

'I could not find matching project information.'

SQL Results:

{results}

User Question:

{user_query}"""


def _query_database(user_query: str) -> pd.DataFrame:
    """Select and execute the SQL helper most relevant to the user query.

    Args:
        user_query: User's project-related question.

    Returns:
        DataFrame returned by the selected SQL helper.
    """

    normalized_query = user_query
    spelling_corrections = {
        "plateform": "Platform",
        "platfrom": "Platform",
    }
    for misspelling, correction in spelling_corrections.items():
        normalized_query = " ".join(
            correction if word.lower() == misspelling else word
            for word in normalized_query.split()
        )

    query_lower = normalized_query.lower()

    if "name" in query_lower or "intern" in query_lower:
        return find_intern_by_name(user_query)

    if (
        "project" in query_lower
        or "working on" in query_lower
        or "assigned to" in query_lower
        or "platform" in query_lower
        or "module" in query_lower
        or "project name" in query_lower
    ):
        common_words = {
            "who",
            "what",
            "is",
            "working",
            "on",
            "assigned",
            "to",
            "the",
            "project",
            "?",
        }
        cleaned_project = " ".join(
            word
            for word in normalized_query.replace("?", " ").split()
            if word.lower() not in common_words
        )
        return find_by_project(cleaned_project)

    if "mentor" in query_lower:
        return find_by_mentor(user_query)

    if "completed" in query_lower:
        return find_by_status("Completed")

    if "active" in query_lower:
        return find_by_status("Active")

    if "status" in query_lower:
        return find_by_status(user_query)

    return show_all_interns()


def project_agent(user_query: str) -> str:
    """Answer a project-related question using SQLite results and Gemini.

    Args:
        user_query: User's project-related question.

    Returns:
        Gemini's answer grounded only in the selected SQL query results.

    Raises:
        ValueError: If ``user_query`` is empty.
        RuntimeError: If SQL lookup or Gemini generation fails.
    """

    cleaned_query = user_query.strip()
    if not cleaned_query:
        raise ValueError("user_query must not be empty.")

    try:
        dataframe = _query_database(cleaned_query)
        results = _dataframe_to_text(dataframe)
    except Exception as exc:
        raise RuntimeError("Failed to retrieve project data from SQLite.") from exc

    prompt = _build_prompt(results=results, user_query=cleaned_query)

    try:
        llm = _load_llm()
        response = llm.invoke(prompt)
    except Exception as exc:
        print(exc)
        raise RuntimeError("Failed to generate Project answer with Gemini.") from exc
        
    answer = getattr(response, "content", "")
    if isinstance(answer, str):
        return answer.strip()

    return str(answer).strip()


def main() -> None:
    """Run a simple Project Agent smoke test."""

    test_query = "Who is working on Knowledge Graph Platform?"

    try:
        print(project_agent(test_query))
    except Exception as exc:
        print(f"Project agent failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
