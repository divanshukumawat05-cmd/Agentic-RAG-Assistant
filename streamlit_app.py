"""Streamlit chat interface for the Agentic RAG Assistant."""

import sys
from pathlib import Path
from typing import Dict, List

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app import process_query  # noqa: E402


Message = Dict[str, str]


def configure_page() -> None:
    """Configure the Streamlit page metadata and base layout."""
    st.set_page_config(
        page_title="Agentic RAG Assistant",
        page_icon="💬",
        layout="centered",
    )


def render_sidebar() -> None:
    """Render project metadata and feature highlights in the sidebar."""
    with st.sidebar:
        st.header("Project Name")
        st.write("Agentic RAG Assistant")

        st.header("Features")
        st.write("✅ HR Agent")
        st.write("✅ Technical Agent")
        st.write("✅ Project Agent")
        st.write("✅ ChromaDB")
        st.write("✅ SQLite")
        st.write("✅ Gemini 2.5 Flash")


def initialize_chat_history() -> None:
    """Create chat history storage in session state when absent."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def get_chat_history() -> List[Message]:
    """Return the current chat history from Streamlit session state."""
    return st.session_state.messages


def render_chat_history(messages: List[Message]) -> None:
    """Display all previous chat messages."""
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def append_message(role: str, content: str) -> None:
    """Append a chat message to the current session history."""
    st.session_state.messages.append({"role": role, "content": content})


def generate_assistant_response(user_query: str) -> str:
    """Generate an assistant response for the submitted user query."""
    try:
        return process_query(user_query)
    except Exception as error:
        return f"An error occurred:\n{error}"


def main() -> None:
    """Run the Streamlit chat application."""
    configure_page()
    render_sidebar()
    initialize_chat_history()

    st.title("Agentic RAG Assistant")
    st.caption("Ask HR, Technical or Project related questions.")

    messages = get_chat_history()
    render_chat_history(messages)

    user_query = st.chat_input("Ask a question")
    if not user_query:
        return

    append_message("user", user_query)
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            assistant_response = generate_assistant_response(user_query)
        st.markdown(assistant_response)

    append_message("assistant", assistant_response)


if __name__ == "__main__":
    main()
