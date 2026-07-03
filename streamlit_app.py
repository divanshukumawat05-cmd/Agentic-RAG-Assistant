"""Streamlit chat interface for the Agentic RAG Assistant."""

from __future__ import annotations

import html
import re
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app import process_query  # noqa: E402
from router.router import route_query  # noqa: E402


Message = Dict[str, str]

QUICK_PROMPTS = [
    "How many leave days are allowed?",
    "How do I install Docker?",
    "Who is working on Knowledge Graph Platform?",
    "What is the internship period?",
]

AGENT_BADGES = {
    "hr": "🟢 HR Agent",
    "technical": "🔵 Technical Agent",
    "project": "🟣 Project Agent",
    "assistant": "🤖 AI Assistant",
}

AGENT_CATEGORY_MAPPING = {
    "HR": "hr",
    "TECHNICAL": "technical",
    "PROJECT": "project",
}

AGENT_SUGGESTED_QUESTIONS = {
    "🟢 HR Agent": [
        "What is the attendance policy?",
        "Is work from home allowed?",
        "What is the internship duration?",
    ],
    "🔵 Technical Agent": [
        "How do I configure Docker?",
        "How do I install Python?",
        "What are the project prerequisites?",
    ],
    "🟣 Project Agent": [
        "Who is assigned to the Payment Module?",
        "Show completed interns",
        "Who is the mentor for this project?",
    ],
}

THINKING_ROUTER_STEPS = [
    "🧠 Understanding your question...",
    "🔀 Router analyzing...",
]
THINKING_AGENT_STEPS = {
    "hr": "🟢 HR Agent selected",
    "technical": "🔵 Technical Agent selected",
    "project": "🟣 Project Agent selected",
}
THINKING_COMPLETION_STEPS = [
    "📚 Searching knowledge base...",
    "🤖 Generating response...",
    "✅ Response ready",
]
THINKING_STEP_DELAY_SECONDS = 0.4
ASSISTANT_STREAM_SHORT_DELAY_SECONDS = 0.005
ASSISTANT_STREAM_MEDIUM_DELAY_SECONDS = 0.018
ASSISTANT_STREAM_LONG_DELAY_SECONDS = 0.012
ASSISTANT_STREAM_MAX_SECONDS = 3.0
ASSISTANT_STREAM_CURSOR = "▌"


def render_html(markup: str) -> None:
    """Render custom HTML safely through Streamlit."""
    st.markdown(textwrap.dedent(markup).strip(), unsafe_allow_html=True)


def configure_page() -> None:
    """Configure the Streamlit page metadata and base layout."""
    st.set_page_config(
        page_title="Agentic RAG Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_custom_css() -> None:
    """Inject the premium dark theme and chat styling."""
    render_html(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

            :root {
                --bg: #0B0F19;
                --card: #151A28;
                --card-soft: rgba(21, 26, 40, 0.76);
                --accent: #4F8CFF;
                --accent-2: #00E5FF;
                --text: #FFFFFF;
                --muted: rgba(255, 255, 255, 0.66);
                --border: rgba(79, 140, 255, 0.34);
                --glow: 0 0 26px rgba(79, 140, 255, 0.24);
            }

            html {
                scroll-behavior: smooth;
            }

            body,
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(79, 140, 255, 0.18), transparent 34rem),
                    radial-gradient(circle at top right, rgba(0, 229, 255, 0.12), transparent 30rem),
                    var(--bg);
                color: var(--text);
                font-family: 'Inter', sans-serif;
            }

            .block-container {
                max-width: 1060px;
                padding: 3rem 2rem 2rem;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, rgba(15, 21, 36, 0.96), rgba(11, 15, 25, 0.98));
                border-right: 1px solid var(--border);
                box-shadow: 18px 0 45px rgba(0, 0, 0, 0.28);
            }

            [data-testid="stSidebar"] * {
                color: var(--text);
                font-family: 'Inter', sans-serif;
            }

            .sidebar-shell {
                display: flex;
                min-height: calc(100vh - 4rem);
                flex-direction: column;
                gap: 1.1rem;
                padding-top: 0.4rem;
            }

            .brand-panel,
            .side-panel,
            .welcome-panel,
            .chat-panel,
            .input-panel {
                border: 1px solid var(--border);
                background: linear-gradient(145deg, rgba(21, 26, 40, 0.84), rgba(21, 26, 40, 0.48));
                box-shadow: var(--glow), inset 0 1px 0 rgba(255, 255, 255, 0.06);
                backdrop-filter: blur(18px);
                border-radius: 22px;
            }

            .brand-panel {
                padding: 1.1rem;
            }

            .brand-title {
                font-size: 1.08rem;
                font-weight: 800;
                line-height: 1.3;
            }

            .status-pill {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                margin-top: 0.9rem;
                padding: 0.45rem 0.7rem;
                border: 1px solid rgba(0, 229, 255, 0.32);
                border-radius: 999px;
                background: rgba(0, 229, 255, 0.08);
                color: rgba(255, 255, 255, 0.86);
                font-size: 0.86rem;
                font-weight: 600;
            }

            .side-panel {
                padding: 1rem;
            }

            .side-heading {
                margin-bottom: 0.75rem;
                color: rgba(255, 255, 255, 0.58);
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }

            .side-item {
                display: flex;
                align-items: center;
                gap: 0.64rem;
                padding: 0.58rem 0;
                color: rgba(255, 255, 255, 0.84);
                font-size: 0.92rem;
                font-weight: 600;
            }

            .current-agent-card {
                animation: sidebarAgentFade 260ms ease-out both;
            }

            .current-agent-active {
                display: flex;
                align-items: center;
                gap: 0.64rem;
                padding: 0.72rem 0;
                color: #FFFFFF;
                font-size: 1rem;
                font-weight: 800;
            }

            .current-agent-muted {
                display: flex;
                align-items: center;
                gap: 0.64rem;
                padding: 0.5rem 0;
                color: rgba(255, 255, 255, 0.42);
                font-size: 0.9rem;
                font-weight: 650;
            }

            .current-agent-divider {
                height: 1px;
                margin: 0.35rem 0;
                background: rgba(255, 255, 255, 0.14);
            }

            @keyframes sidebarAgentFade {
                from {
                    opacity: 0;
                    transform: translateY(4px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .sidebar-footer {
                margin-top: auto;
                padding: 0.9rem 0.25rem 0;
                color: rgba(255, 255, 255, 0.55);
                font-size: 0.86rem;
                text-align: center;
            }

            .hero {
                margin: 0 auto 1.8rem;
                text-align: center;
            }

            .hero h1 {
                margin: 0;
                background: linear-gradient(92deg, #FFFFFF 8%, var(--accent) 48%, var(--accent-2) 92%);
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
                font-size: clamp(2.45rem, 5vw, 4.6rem);
                font-weight: 800;
                letter-spacing: 0;
                line-height: 1.05;
            }

            .hero p {
                max-width: 760px;
                margin: 0.9rem auto 0;
                color: var(--muted);
                font-size: 1.05rem;
                line-height: 1.7;
            }

            .chat-panel {
                min-height: 420px;
                padding: 1.1rem;
            }

            .message-row {
                display: flex;
                align-items: flex-end;
                gap: 10px;
                margin: 0 0 20px;
                animation: messageFadeIn 220ms ease-out both;
            }

            .message-row.user {
                justify-content: flex-end;
            }

            .message-row.assistant {
                justify-content: flex-start;
                animation: assistantMessageIn 260ms ease-out both;
            }

            .message-avatar {
                display: grid;
                width: 28px;
                height: 28px;
                flex: 0 0 28px;
                place-items: center;
                border: 1px solid rgba(255, 255, 255, 0.14);
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.08);
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.18);
                font-size: 0.9rem;
                line-height: 1;
            }

            .message-card {
                display: inline-flex;
                flex-direction: column;
                flex: 0 1 auto;
                width: auto;
                height: auto;
                box-sizing: border-box;
                border-radius: 18px;
                padding: 10px 14px;
                line-height: 1.55;
                white-space: pre-wrap;
                overflow-wrap: anywhere;
                word-break: normal;
            }

            .message-card.user {
                min-width: 90px;
                max-width: 55%;
                border: 1px solid rgba(79, 140, 255, 0.48);
                background: linear-gradient(135deg, rgba(21, 55, 112, 0.96), rgba(17, 37, 78, 0.94));
                box-shadow: 0 12px 32px rgba(79, 140, 255, 0.16);
            }

            .message-card.assistant {
                max-width: 60%;
                border: 1px solid rgba(0, 229, 255, 0.22);
                background: linear-gradient(145deg, rgba(21, 26, 40, 0.72), rgba(255, 255, 255, 0.05));
                box-shadow: 0 18px 42px rgba(0, 0, 0, 0.24), inset 0 1px 0 rgba(255, 255, 255, 0.06);
                backdrop-filter: blur(18px);
            }

            .message-text {
                color: #FFFFFF;
            }

            .stream-cursor {
                display: inline-block;
                margin-left: 0.12rem;
                color: #FFFFFF;
                animation: streamCursorBlink 900ms steps(2, start) infinite;
            }

            .message-timestamp {
                align-self: flex-end;
                margin-top: 0.34rem;
                color: rgba(255, 255, 255, 0.52);
                font-size: 0.68rem;
                font-weight: 700;
                line-height: 1;
            }

            .assistant-footer {
                display: grid;
                gap: 0.26rem;
                margin-top: 0.7rem;
                padding-top: 0.56rem;
                border-top: 1px solid rgba(255, 255, 255, 0.13);
                color: rgba(255, 255, 255, 0.52);
                font-size: 0.68rem;
                line-height: 1.35;
            }

            .assistant-footer-row {
                display: grid;
                grid-template-columns: minmax(112px, auto) 1fr;
                gap: 0.9rem;
                align-items: center;
            }

            .assistant-footer-label {
                white-space: nowrap;
            }

            .assistant-footer-value {
                justify-self: end;
                color: rgba(255, 255, 255, 0.66);
                font-weight: 700;
                text-align: right;
            }

            .agent-badge {
                display: inline-flex;
                align-items: center;
                align-self: flex-start;
                padding: 0.28rem 0.56rem;
                margin-bottom: 0.42rem;
                border: 1px solid rgba(79, 140, 255, 0.36);
                border-radius: 999px;
                background: rgba(79, 140, 255, 0.12);
                color: #FFFFFF;
                font-size: 0.72rem;
                font-weight: 800;
            }

            @keyframes messageFadeIn {
                from {
                    opacity: 0;
                }
                to {
                    opacity: 1;
                }
            }

            @keyframes assistantMessageIn {
                from {
                    opacity: 0;
                    transform: translateY(6px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            @keyframes streamCursorBlink {
                0%,
                45% {
                    opacity: 1;
                }
                46%,
                100% {
                    opacity: 0;
                }
            }

            .thinking-timeline {
                display: flex;
                justify-content: center;
                margin: 1rem 0;
                animation: thinkingFadeIn 260ms ease-out both;
            }

            .thinking-card {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-width: 260px;
                max-width: min(92%, 520px);
                padding: 0.78rem 1.05rem;
                border: 1px solid rgba(79, 140, 255, 0.32);
                border-radius: 18px;
                background: linear-gradient(145deg, rgba(21, 26, 40, 0.74), rgba(255, 255, 255, 0.05));
                box-shadow: 0 0 24px rgba(79, 140, 255, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.06);
                backdrop-filter: blur(18px);
                color: rgba(255, 255, 255, 0.9);
                font-size: 0.92rem;
                font-weight: 700;
                text-align: center;
            }

            @keyframes thinkingFadeIn {
                from {
                    opacity: 0;
                    transform: translateY(4px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .welcome-panel {
                margin: 0 auto 1rem;
                padding: 1.5rem;
            }

            .welcome-panel h2 {
                margin: 0 0 0.7rem;
                color: #FFFFFF;
                font-size: 1.55rem;
                font-weight: 800;
                letter-spacing: 0;
            }

            .welcome-panel p,
            .welcome-panel li {
                color: rgba(255, 255, 255, 0.74);
                font-size: 0.98rem;
            }

            .prompt-grid-title {
                margin: 1.2rem 0 0.65rem;
                color: rgba(255, 255, 255, 0.58);
                font-size: 0.76rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }

            .input-panel {
                margin-top: 1rem;
                padding: 0.8rem;
            }

            [data-testid="stForm"] {
                margin-top: 1rem;
                padding: 0.8rem;
                border: 1px solid var(--border);
                border-radius: 22px;
                background: linear-gradient(145deg, rgba(21, 26, 40, 0.84), rgba(21, 26, 40, 0.48));
                box-shadow: var(--glow), inset 0 1px 0 rgba(255, 255, 255, 0.06);
                backdrop-filter: blur(18px);
            }

            div[data-testid="stTextInput"] input {
                min-height: 54px;
                border: 1px solid rgba(79, 140, 255, 0.34);
                border-radius: 18px;
                background: rgba(11, 15, 25, 0.82);
                color: #FFFFFF;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
                font-size: 0.98rem;
            }

            div[data-testid="stTextInput"] input:focus {
                border-color: var(--accent-2);
                box-shadow: 0 0 0 1px rgba(0, 229, 255, 0.28), 0 0 22px rgba(0, 229, 255, 0.14);
            }

            .stButton > button,
            .stFormSubmitButton > button {
                min-height: 46px;
                border: 1px solid rgba(79, 140, 255, 0.42);
                border-radius: 18px;
                background: linear-gradient(135deg, rgba(79, 140, 255, 0.95), rgba(0, 229, 255, 0.78));
                color: #FFFFFF;
                box-shadow: 0 12px 28px rgba(79, 140, 255, 0.22);
                font-weight: 800;
                transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
            }

            .stButton > button:hover,
            .stFormSubmitButton > button:hover {
                border-color: rgba(0, 229, 255, 0.85);
                box-shadow: 0 16px 38px rgba(0, 229, 255, 0.24);
                transform: translateY(-1px);
            }

            .stButton > button:active,
            .stFormSubmitButton > button:active {
                transform: translateY(0);
            }

            [data-testid="stSpinner"] {
                color: #FFFFFF;
            }

            header[data-testid="stHeader"],
            [data-testid="stToolbar"] {
                background: transparent;
            }

            .suggested-questions-section {
                margin-top: 1.5rem;
                margin-bottom: 0;
                animation: suggestionsSlideIn 300ms ease-out both;
            }

            .suggested-questions-title {
                margin-bottom: 0.9rem;
                color: rgba(255, 255, 255, 0.72);
                font-size: 0.95rem;
                font-weight: 800;
                letter-spacing: 0.05em;
            }

            .suggested-questions-grid {
                display: flex;
                flex-wrap: wrap;
                gap: 0.8rem;
                margin-bottom: 1rem;
            }

            .suggestion-button-wrapper {
                display: inline-block;
            }

            .suggestion-button-wrapper .stButton > button {
                padding: 0.5rem 0.875rem;
                font-size: 0.8rem;
                font-weight: 700;
                border: 1px solid rgba(79, 140, 255, 0.35);
                border-radius: 999px;
                background: linear-gradient(135deg, rgba(79, 140, 255, 0.16), rgba(0, 229, 255, 0.08));
                box-shadow: 0 4px 12px rgba(79, 140, 255, 0.14), inset 0 1px 0 rgba(255, 255, 255, 0.08);
                backdrop-filter: blur(12px);
                color: rgba(255, 255, 255, 0.92);
                transition: all 220ms cubic-bezier(0.4, 0, 0.2, 1);
                white-space: nowrap;
            }

            .suggestion-button-wrapper .stButton > button:hover {
                border-color: rgba(0, 229, 255, 0.58);
                background: linear-gradient(135deg, rgba(79, 140, 255, 0.24), rgba(0, 229, 255, 0.14));
                box-shadow: 0 6px 20px rgba(79, 140, 255, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.1);
                transform: scale(1.04);
                color: #FFFFFF;
            }

            .suggestion-button-wrapper .stButton > button:active {
                transform: scale(0.98);
            }

            @keyframes suggestionsSlideIn {
                from {
                    opacity: 0;
                    transform: translateY(8px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .router-visualization {
                margin: 1.2rem 0;
                animation: routerFadeIn 300ms ease-out both;
            }

            .router-glass-card {
                display: inline-flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-width: 340px;
                max-width: 520px;
                padding: 1.4rem;
                border: 1px solid rgba(79, 140, 255, 0.28);
                border-radius: 20px;
                background: linear-gradient(145deg, rgba(21, 26, 40, 0.72), rgba(255, 255, 255, 0.05));
                box-shadow: 0 12px 40px rgba(79, 140, 255, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.06);
                backdrop-filter: blur(18px);
                gap: 1rem;
            }

            .router-step {
                display: flex;
                align-items: center;
                gap: 0.8rem;
                width: 100%;
                padding: 0.65rem 0.8rem;
                border-radius: 12px;
                transition: all 200ms ease-out;
            }

            .router-step.completed {
                background: rgba(0, 229, 255, 0.08);
            }

            .router-step.active {
                background: rgba(79, 140, 255, 0.16);
            }

            .router-step.pending {
                opacity: 0.6;
            }

            .router-step-icon {
                display: flex;
                align-items: center;
                justify-content: center;
                width: 24px;
                height: 24px;
                flex-shrink: 0;
                font-size: 0.9rem;
                font-weight: 800;
            }

            .router-step-icon.completed {
                color: #00E5FF;
            }

            .router-step-icon.active {
                color: #4F8CFF;
                animation: stepPulse 1.2s ease-in-out infinite;
            }

            .router-step-icon.pending {
                color: rgba(255, 255, 255, 0.35);
            }

            .router-step-text {
                flex: 1;
                color: rgba(255, 255, 255, 0.88);
                font-size: 0.92rem;
                font-weight: 700;
            }

            .router-step.completed .router-step-text {
                color: rgba(255, 255, 255, 0.7);
            }

            .router-step.active .router-step-text {
                color: #FFFFFF;
            }

            @keyframes stepPulse {
                0%, 100% {
                    box-shadow: 0 0 8px rgba(79, 140, 255, 0.6);
                    opacity: 1;
                }
                50% {
                    box-shadow: 0 0 16px rgba(79, 140, 255, 0.3);
                    opacity: 0.85;
                }
            }

            @keyframes routerFadeIn {
                from {
                    opacity: 0;
                    transform: translateY(-8px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            @media (max-width: 760px) {
                .block-container {
                    padding: 2rem 1rem 1rem;
                }

                .message-card {
                    max-width: 85%;
                }

                .router-glass-card {
                    min-width: 280px;
                    padding: 1rem;
                }
            }
        </style>
        """
    )


def get_sidebar_agent_badges() -> List[str]:
    """Return the agent badges shown in the dynamic sidebar."""
    return [
        AGENT_BADGES["hr"],
        AGENT_BADGES["technical"],
        AGENT_BADGES["project"],
    ]


def get_current_sidebar_agent_badge() -> str:
    """Return the latest detected assistant agent badge for the sidebar."""
    agent_badges = set(get_sidebar_agent_badges())
    for message in reversed(st.session_state.get("messages", [])):
        if message.get("role") == "assistant" and message.get("badge") in agent_badges:
            return message["badge"]
    return ""


def render_current_agent_sidebar_html() -> str:
    """Return the dynamic current-agent sidebar panel HTML."""
    active_badge = get_current_sidebar_agent_badge()
    agent_badges = get_sidebar_agent_badges()

    if not active_badge:
        muted_agents = "\n".join(
            f'<div class="current-agent-muted">⚪ {html.escape(get_agent_label(badge))}</div>'
            for badge in agent_badges
        )
        active_content = '<div class="current-agent-muted">Waiting for first response</div>'
    else:
        muted_agents = "\n".join(
            f'<div class="current-agent-muted">⚪ {html.escape(get_agent_label(badge))}</div>'
            for badge in agent_badges
            if badge != active_badge
        )
        active_content = (
            f'<div class="current-agent-active">{html.escape(active_badge)}</div>'
        )

    return textwrap.dedent(
        f"""
        <div class="side-panel current-agent-card">
            <div class="side-heading">🎯 CURRENT AGENT</div>
            {active_content}
            <div class="current-agent-divider"></div>
            {muted_agents}
        </div>
        """
    ).strip()


def render_sidebar() -> None:
    """Render the application sidebar."""
    current_agent_panel = render_current_agent_sidebar_html()

    with st.sidebar:
        render_html(
            f"""
            <div class="sidebar-shell">
                <div class="brand-panel">
                    <div class="brand-title">🤖 Agentic RAG Assistant</div>
                    <div class="status-pill">🟢 Online</div>
                </div>
                {current_agent_panel}
                <div class="side-panel">
                    <div class="side-heading">Technology Stack</div>
                    <div class="side-item">Gemini 2.5 Flash</div>
                    <div class="side-item">LangChain</div>
                    <div class="side-item">ChromaDB</div>
                    <div class="side-item">SQLite</div>
                    <div class="side-item">Python</div>
                </div>
                <div class="sidebar-footer">Made by Divanshu Kumawat</div>
            </div>
            """
        )


def initialize_chat_history() -> None:
    """Create chat history storage in session state when absent."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def get_chat_history() -> List[Message]:
    """Return the current chat history from Streamlit session state."""
    return st.session_state.messages


def current_timestamp() -> str:
    """Return a compact timestamp for chat messages."""
    return datetime.now().strftime("%I:%M %p").lstrip("0")


def detect_agent_badge(user_query: str, assistant_response: str = "") -> str:
    """Select a display badge from UI-visible query and response text."""
    text = f"{user_query} {assistant_response}".lower()

    hr_terms = {
        "leave",
        "attendance",
        "holiday",
        "stipend",
        "internship period",
        "policy",
        "working hour",
        "code of conduct",
        "mentor communication",
        "performance evaluation",
    }
    technical_terms = {
        "python",
        "docker",
        "git",
        "api",
        "install",
        "setup",
        "deploy",
        "debug",
        "vs code",
        "programming",
        "technical",
    }
    project_terms = {
        "project",
        "knowledge graph",
        "working on",
        "completion",
        "progress",
        "mentor",
        "assigned",
        "database",
    }

    if any(term in text for term in hr_terms):
        return AGENT_BADGES["hr"]
    if any(term in text for term in technical_terms):
        return AGENT_BADGES["technical"]
    if any(term in text for term in project_terms):
        return AGENT_BADGES["project"]

    return AGENT_BADGES["assistant"]


def append_message(
    role: str,
    content: str,
    *,
    badge: str = "",
    timestamp: str | None = None,
    response_time: str = "",
) -> None:
    """Append a chat message to the current session history."""
    st.session_state.messages.append(
        {
            "role": role,
            "content": content,
            "badge": badge,
            "timestamp": timestamp or current_timestamp(),
            "response_time": response_time,
        }
    )


def get_agent_label(badge: str) -> str:
    """Return a plain agent label from the UI badge."""
    if "HR Agent" in badge:
        return "HR Agent"
    if "Technical Agent" in badge:
        return "Technical Agent"
    if "Project Agent" in badge:
        return "Project Agent"
    return "AI Assistant"


def render_assistant_footer(message: Message) -> str:
    """Return metadata footer HTML for an assistant message."""
    response_time = html.escape(message.get("response_time", ""))
    if not response_time:
        return ""

    agent_label = html.escape(get_agent_label(message.get("badge", "")))
    generated_time = html.escape(message.get("timestamp", ""))

    return textwrap.dedent(
        f"""
        <div class="assistant-footer">
            <div class="assistant-footer-row">
                <span class="assistant-footer-label">⚡ Response Time</span>
                <span class="assistant-footer-value">{response_time}</span>
            </div>
            <div class="assistant-footer-row">
                <span class="assistant-footer-label">🧠 Agent</span>
                <span class="assistant-footer-value">{agent_label}</span>
            </div>
            <div class="assistant-footer-row">
                <span class="assistant-footer-label">🕒 Generated</span>
                <span class="assistant-footer-value">{generated_time}</span>
            </div>
        </div>
        """
    ).strip()


def render_message_html(message: Message) -> str:
    """Return the HTML for one chat bubble."""
    role = message["role"]
    content = html.escape(message["content"])
    timestamp = html.escape(message.get("timestamp", ""))
    badge = html.escape(message.get("badge", AGENT_BADGES["assistant"]))
    is_streaming = message.get("streaming", "") == "true"

    if role == "user":
        avatar = '<div class="message-avatar user-avatar" aria-hidden="true">👤</div>'
        bubble = textwrap.dedent(
            f"""
            <div class="message-card {role}">
                <div class="message-text">{content}</div>
                <div class="message-timestamp">{timestamp}</div>
            </div>
            {avatar}
            """
        ).strip()
    else:
        if is_streaming:
            content = f'{content}<span class="stream-cursor">{ASSISTANT_STREAM_CURSOR}</span>'
        avatar = '<div class="message-avatar assistant-avatar" aria-hidden="true">🤖</div>'
        footer = render_assistant_footer(message)
        bubble = textwrap.dedent(
            f"""
            {avatar}
            <div class="message-card {role}">
                <div class="agent-badge">{badge}</div>
                <div class="message-text">{content}</div>
                {footer}
            </div>
            """
        ).strip()

    return textwrap.dedent(
        f"""
        <div class="message-row {role}">
            {bubble}
        </div>
        """
    ).strip()


def render_chat_history(messages: List[Message]) -> None:
    """Display all previous chat messages."""
    rendered_messages = ['<div class="chat-panel">']
    for message in messages:
        rendered_messages.append(render_message_html(message))
    rendered_messages.append("</div>")

    render_html("\n".join(rendered_messages))


def render_hero() -> None:
    """Render the main page title."""
    render_html(
        """
        <section class="hero">
            <h1>Agentic RAG Assistant</h1>
            <p>Multi-Agent AI Assistant powered by Gemini, LangChain, ChromaDB and SQLite.</p>
        </section>
        """
    )


def render_welcome_screen() -> str:
    """Render welcome content and quick prompt buttons."""
    selected_prompt = ""

    render_html(
        """
        <div class="welcome-panel">
            <h2>👋 Welcome</h2>
            <p>Ask anything about:</p>
            <ul>
                <li>HR Policies</li>
                <li>Technical Documentation</li>
                <li>Internship Projects</li>
            </ul>
            <div class="prompt-grid-title">Quick Prompts</div>
        </div>
        """
    )

    cols = st.columns(2)
    for index, prompt in enumerate(QUICK_PROMPTS):
        with cols[index % 2]:
            if st.button(prompt, key=f"quick_prompt_{index}", use_container_width=True):
                selected_prompt = prompt

    return selected_prompt


def render_chat_input() -> str:
    """Render the rounded chat input and send button."""
    with st.form("chat_form", clear_on_submit=True):
        input_col, button_col = st.columns([5, 1])
        with input_col:
            user_query = st.text_input(
                "Message",
                placeholder="Ask HR, Technical or Project related questions...",
                label_visibility="collapsed",
            )
        with button_col:
            submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted:
        return user_query.strip()
    return ""


def render_suggested_questions(agent_badge: str) -> str:
    """Render suggested follow-up questions based on the active agent."""
    suggestions = AGENT_SUGGESTED_QUESTIONS.get(agent_badge, [])
    if not suggestions:
        return ""

    render_html(
        """
        <div class="suggested-questions-section">
            <div class="suggested-questions-title">💡 Suggested Questions</div>
            <div class="suggested-questions-grid">
        """
    )

    for idx, suggestion in enumerate(suggestions):
        st.markdown(
            f'<div class="suggestion-button-wrapper">',
            unsafe_allow_html=True
        )
        if st.button(
            suggestion,
            key=f"suggestion_btn_{agent_badge}_{idx}",
            use_container_width=False
        ):
            st.markdown("</div>", unsafe_allow_html=True)
            render_html("</div></div>")
            return suggestion
        st.markdown("</div>", unsafe_allow_html=True)

    render_html("</div></div>")
    return ""


def generate_assistant_response(user_query: str) -> str:
    """Generate an assistant response for the submitted user query."""
    assistant_response, _response_time_seconds, _agent_category = generate_timed_assistant_response_with_routing(user_query)
    return assistant_response


def generate_timed_assistant_response_with_routing(user_query: str) -> Tuple[str, float, str]:
    """Generate an assistant response with accurate backend execution time and routing category."""
    agent_category = ""
    assistant_response = ""
    response_time_seconds = 0.0

    try:
        # Get routing category for thinking timeline (not included in response time)
        agent_category = route_query(user_query.strip())

        # Measure ONLY the actual backend execution time
        start_time = time.perf_counter()
        assistant_response = process_query(user_query)
        response_time_seconds = time.perf_counter() - start_time
    except Exception as error:
        assistant_response = f"An error occurred:\n{error}"

    return assistant_response, response_time_seconds, agent_category


def render_thinking_status(container, status: str) -> None:
    """Render the current AI thinking timeline status."""
    safe_status = html.escape(status)
    container.markdown(
        textwrap.dedent(
            f"""
            <div class="thinking-timeline">
                <div class="thinking-card">{safe_status}</div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


def build_router_visualization_html(steps: List[Tuple[str, str]]) -> str:
    """Build HTML for router visualization with all workflow steps."""
    steps_html = []
    for text, status in steps:
        safe_text = html.escape(text)
        
        if status == "completed":
            icon = "✔"
            step_class = "completed"
        elif status == "active":
            icon = "◉"
            step_class = "active"
        else:
            icon = "○"
            step_class = "pending"
        
        steps_html.append(
            f'<div class="router-step {step_class}">'
            f'<div class="router-step-icon {step_class}">{icon}</div>'
            f'<div class="router-step-text">{safe_text}</div>'
            f'</div>'
        )
    
    return (
        '<div class="router-visualization">'
        '<div class="router-glass-card">'
        + "".join(steps_html) +
        '</div></div>'
    )


def render_router_visualization(container, steps: List[Tuple[str, str]]) -> None:
    """Render the multi-agent router visualization."""
    html_content = build_router_visualization_html(steps)
    container.markdown(html_content, unsafe_allow_html=True)


def generate_response_with_thinking_timeline(user_query: str) -> Tuple[str, float, str]:
    """Show professional multi-agent router visualization while the response is generated."""
    visualization = st.empty()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(generate_timed_assistant_response_with_routing, user_query)
        step_index = 0

        # Initial steps (understanding and router analyzing)
        while not future.done() and step_index < 2:
            steps = [
                ("🧠 Understanding your question...", "completed" if step_index > 0 else "active"),
                ("🔀 Router analyzing...", "active" if step_index == 1 else "pending"),
                ("", "pending"),
                ("📚 Searching knowledge base...", "pending"),
                ("🤖 Generating response...", "pending"),
                ("✅ Response ready", "pending"),
            ]
            render_router_visualization(visualization, [(text, status) for text, status in steps if text])
            time.sleep(THINKING_STEP_DELAY_SECONDS)
            step_index += 1

        # Wait for response with agent category
        assistant_response, response_time_seconds, agent_category = future.result()
        
        # Get the agent step text
        agent_key = AGENT_CATEGORY_MAPPING.get(agent_category, "assistant")
        agent_step_text = THINKING_AGENT_STEPS.get(agent_key, "🤖 AI Agent selected")
        
        # Show agent selection step
        steps = [
            ("🧠 Understanding your question...", "completed"),
            ("🔀 Router analyzing...", "completed"),
            (agent_step_text, "active"),
            ("📚 Searching knowledge base...", "pending"),
            ("🤖 Generating response...", "pending"),
            ("✅ Response ready", "pending"),
        ]
        render_router_visualization(visualization, steps)
        time.sleep(THINKING_STEP_DELAY_SECONDS)
        
        # Show searching knowledge base step
        steps[2] = (agent_step_text, "completed")
        steps[3] = ("📚 Searching knowledge base...", "active")
        render_router_visualization(visualization, steps)
        time.sleep(THINKING_STEP_DELAY_SECONDS)
        
        # Show generating response step
        steps[3] = ("📚 Searching knowledge base...", "completed")
        steps[4] = ("🤖 Generating response...", "active")
        render_router_visualization(visualization, steps)
        time.sleep(THINKING_STEP_DELAY_SECONDS)
        
        # Show response ready (final step)
        steps[4] = ("🤖 Generating response...", "completed")
        steps[5] = ("✅ Response ready", "active")
        render_router_visualization(visualization, steps)
        time.sleep(THINKING_STEP_DELAY_SECONDS)

    visualization.empty()
    return assistant_response, response_time_seconds, agent_category


def split_response_words(response: str) -> List[str]:
    """Split a response into word tokens while preserving trailing spaces."""
    return re.findall(r"\S+\s*", response)


def get_stream_chunk_size(word_count: int) -> int:
    """Return an adaptive number of words to reveal per streaming update."""
    if word_count <= 20:
        return 5
    if word_count <= 80:
        return 4
    if word_count <= 160:
        return 8
    if word_count <= 320:
        return 12
    return max(16, word_count // 120)


def get_stream_delay_seconds(word_count: int, update_count: int) -> float:
    """Return an adaptive delay that keeps streaming smooth but fast."""
    if update_count <= 1:
        return 0.0

    if word_count <= 20:
        base_delay = ASSISTANT_STREAM_SHORT_DELAY_SECONDS
    elif word_count <= 160:
        base_delay = ASSISTANT_STREAM_MEDIUM_DELAY_SECONDS
    else:
        base_delay = ASSISTANT_STREAM_LONG_DELAY_SECONDS

    return min(base_delay, ASSISTANT_STREAM_MAX_SECONDS / update_count)


def build_response_stream_chunks(response: str) -> List[str]:
    """Build cumulative response chunks for efficient assistant streaming."""
    words = split_response_words(response)
    if not words:
        return [response]

    chunk_size = get_stream_chunk_size(len(words))
    chunks = []
    for index in range(chunk_size, len(words), chunk_size):
        chunks.append("".join(words[:index]))
    chunks.append(response)
    return chunks


def stream_assistant_response(
    response: str,
    *,
    badge: str,
    timestamp: str,
    response_time: str,
) -> None:
    """Render the real assistant response progressively in one placeholder."""
    stream_placeholder = st.empty()
    stream_chunks = build_response_stream_chunks(response)
    stream_delay = get_stream_delay_seconds(
        len(split_response_words(response)),
        len(stream_chunks),
    )

    for partial_response in stream_chunks:
        stream_placeholder.markdown(
            render_message_html(
                {
                    "role": "assistant",
                    "content": partial_response,
                    "badge": badge,
                    "timestamp": timestamp,
                    "response_time": response_time,
                    "streaming": "true",
                }
            ),
            unsafe_allow_html=True,
        )
        if stream_delay:
            time.sleep(stream_delay)

    stream_placeholder.markdown(
        render_message_html(
            {
                "role": "assistant",
                "content": response,
                "badge": badge,
                "timestamp": timestamp,
                "response_time": response_time,
            }
        ),
        unsafe_allow_html=True,
    )


def handle_user_query(user_query: str) -> None:
    """Append a user query, call the assistant, and append the response."""
    append_message("user", user_query)

    assistant_response, response_time_seconds, agent_category = generate_response_with_thinking_timeline(user_query)
    
    agent_badge = AGENT_BADGES.get(
        AGENT_CATEGORY_MAPPING.get(agent_category, "assistant"),
        AGENT_BADGES["assistant"]
    )
    
    assistant_timestamp = current_timestamp()
    assistant_response_time = f"{response_time_seconds:.2f} sec"

    stream_assistant_response(
        assistant_response,
        badge=agent_badge,
        timestamp=assistant_timestamp,
        response_time=assistant_response_time,
    )

    append_message(
        "assistant",
        assistant_response,
        badge=agent_badge,
        timestamp=assistant_timestamp,
        response_time=assistant_response_time,
    )
    st.rerun()


def main() -> None:
    """Run the Streamlit chat application."""
    configure_page()
    inject_custom_css()
    render_sidebar()
    initialize_chat_history()

    render_hero()

    messages = get_chat_history()
    selected_prompt = ""
    selected_suggestion = ""
    if not messages:
        selected_prompt = render_welcome_screen()

    if messages:
        render_chat_history(messages)

        # Render suggestions after the last assistant message
        if messages[-1].get("role") == "assistant":
            agent_badge = messages[-1].get("badge", "")
            selected_suggestion = render_suggested_questions(agent_badge)

        # Only render input if no suggestion was selected
        if not selected_suggestion:
            submitted_query = render_chat_input()
        else:
            submitted_query = ""
    else:
        submitted_query = render_chat_input()

    user_query = selected_prompt or selected_suggestion or submitted_query

    if user_query:
        handle_user_query(user_query)

if __name__ == "__main__":
    main()
