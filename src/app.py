"""Command-line entry point for the Agentic RAG Assistant."""

from __future__ import annotations

import sys

from agents.hr_agent import hr_agent
from agents.project_agent import project_agent
from agents.technical_agent import technical_agent
from router.router import route_query


NO_AGENT_MESSAGE = "Sorry, no suitable agent is available."


def process_query(user_query: str) -> str:
    """Route a user query to the appropriate RAG agent.

    Args:
        user_query: User's question.

    Returns:
        The selected agent response, or ``NO_AGENT_MESSAGE`` when no suitable
        agent is available.

    Raises:
        ValueError: If ``user_query`` is empty.
        RuntimeError: If routing or agent execution fails.
    """

    cleaned_query = user_query.strip()
    if not cleaned_query:
        raise ValueError("Please enter a valid question.")

    try:
        category = route_query(cleaned_query)
    except Exception as exc:
        raise RuntimeError("Failed to route the query.") from exc

    if category == "HR":
        return hr_agent(cleaned_query)

    if category == "TECHNICAL":
        return technical_agent(cleaned_query)

    if category == "PROJECT":
        return project_agent(cleaned_query)

    return NO_AGENT_MESSAGE


def main() -> None:
    """Run the Agentic RAG Assistant command-line interface."""

    print("==================================")
    print("Agentic RAG Assistant")
    print("==================================")

    while True:
        try:
            user_query = input("Ask your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_query.lower() in {"exit", "quit"}:
            break

        if not user_query:
            print("Please enter a valid question.")
            continue

        try:
            answer = process_query(user_query)
        except Exception as exc:
            answer = f"An error occurred while processing your request: {exc}"

        print("----------------------------------")
        print("Response:")
        print()
        print(answer)
        print()
        print("----------------------------------")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Application failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
