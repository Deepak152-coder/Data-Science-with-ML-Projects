"""
app.py
======
Deepak AI Assistant — a professional, ChatGPT-style chat interface
built on Streamlit + Groq.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from chatbot import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL_LABEL,
    ChatbotError,
    stream_response,
)
from utils import (
    compute_analytics,
    export_as_json,
    export_as_markdown,
    export_as_pdf,
    export_as_txt,
    import_chat_from_json,
    process_uploaded_file,
)

load_dotenv()

LOCAL_KEY_EXISTS = bool(os.getenv("GROQ_API_KEY"))

DEFAULT_SYSTEM_PROMPT = """You are Deepak AI Assistant, a friendly, intelligent, and supportive AI companion.

Your personality:
- Friendly and approachable
- Helpful without being overly formal
- Patient and encouraging
- Honest when you don't know something
- Conversational and natural

Guidelines:
- Answer questions clearly and accurately.
- Adapt your explanation to the user's level.
- For coding questions, provide working code and explanations.
- For learning topics, teach step by step.
- For brainstorming, be creative and collaborative.
- For personal productivity and career questions, act like a supportive friend and mentor.
- Use examples whenever helpful.
- Keep responses engaging and easy to understand.
- Avoid sounding robotic.

You are not just an information provider; you are a helpful companion who assists with learning,
problem-solving, projects, career growth, and everyday questions."""


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Deepak AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# SESSION STATE INIT
# =========================

def init_session_state() -> None:
    """Initialise all session_state keys used across the app."""
    defaults = {
        "messages": [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "model_label": DEFAULT_MODEL_LABEL,
        "temperature": 0.7,
        "max_tokens": 1024,
        "session_start": datetime.now(),
        "search_query": "",
        "pending_file_context": None,
        "last_response_time": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# =========================
# CUSTOM CSS — GLASSMORPHISM / DARK THEME
# =========================

st.markdown(
    """
<style>

:root {
    --accent-1: #00DBDE;
    --accent-2: #FC00FF;
    --bg-glass: rgba(255, 255, 255, 0.04);
    --border-glass: rgba(255, 255, 255, 0.10);
}

.stApp {
    background: radial-gradient(circle at 20% -10%, #1b1f3a 0%, #0b0d17 45%, #05060b 100%);
}

/* Hide default footer / menu clutter */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }

/* Hero title */
.chat-title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

.subtitle {
    text-align: center;
    color: #9aa0b4;
    margin-top: 4px;
    margin-bottom: 22px;
    font-size: 0.95rem;
}

/* Glass card */
.glass-card {
    background: var(--bg-glass);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 18px 20px;
    backdrop-filter: blur(10px);
    margin-bottom: 14px;
}

/* Welcome screen suggestion cards */
.suggestion-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    margin-top: 10px;
}

.suggestion-card {
    background: var(--bg-glass);
    border: 1px solid var(--border-glass);
    border-radius: 14px;
    padding: 14px 16px;
    backdrop-filter: blur(8px);
    transition: transform 0.15s ease, border-color 0.15s ease;
}

.suggestion-card:hover {
    transform: translateY(-2px);
    border-color: var(--accent-1);
}

.suggestion-title { font-weight: 600; color: #e8e8f0; margin-bottom: 4px; }
.suggestion-sub { color: #8d93a8; font-size: 0.85rem; }

/* Chat bubbles */
[data-testid="stChatMessage"] {
    background: var(--bg-glass);
    border: 1px solid var(--border-glass);
    border-radius: 14px;
    backdrop-filter: blur(8px);
    padding: 4px 6px;
}

.msg-timestamp {
    font-size: 0.72rem;
    color: #6c7287;
    margin-top: 2px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(10, 12, 22, 0.85);
    border-right: 1px solid var(--border-glass);
}

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--bg-glass);
    border: 1px solid var(--border-glass);
    border-radius: 12px;
    padding: 8px 10px;
}

/* Buttons */
.stButton button {
    border-radius: 10px;
    border: 1px solid var(--border-glass);
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================
# HEADER
# =========================

st.markdown('<p class="chat-title">🤖 Deepak AI Assistant</p>', unsafe_allow_html=True)
st.markdown(
    f'<p class="subtitle">Powered by Groq · {st.session_state.model_label}</p>',
    unsafe_allow_html=True,
)


# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.title("⚙️ Control Panel")

    # ---- API KEY ----
    if not LOCAL_KEY_EXISTS:
        api_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            help="Required when running on Streamlit Cloud.",
        )
        st.caption("Get your key from https://console.groq.com/keys")
    else:
        api_key = None
        st.success("✅ Using local .env API key")

    st.divider()

    # ---- MODEL SELECTOR ----
    st.subheader("🧠 Model")
    st.session_state.model_label = st.selectbox(
        "Choose a model",
        options=list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(st.session_state.model_label),
    )

    # ---- GENERATION PARAMS ----
    st.subheader("🎛️ Generation Settings")
    st.session_state.temperature = st.slider(
        "Temperature", min_value=0.0, max_value=1.5,
        value=st.session_state.temperature, step=0.05,
        help="Higher = more creative, lower = more focused.",
    )
    st.session_state.max_tokens = st.slider(
        "Max tokens", min_value=128, max_value=8192,
        value=st.session_state.max_tokens, step=128,
    )

    st.divider()

    # ---- SYSTEM PROMPT EDITOR ----
    with st.expander("📝 System Prompt Editor"):
        new_prompt = st.text_area(
            "Edit the assistant's behaviour",
            value=st.session_state.system_prompt,
            height=180,
        )
        if st.button("Apply System Prompt", use_container_width=True):
            st.session_state.system_prompt = new_prompt
            st.session_state.messages[0] = {"role": "system", "content": new_prompt}
            st.success("System prompt updated.")

    st.divider()

    # ---- FILE UPLOAD ----
    st.subheader("📎 Attach a File")
    uploaded_file = st.file_uploader(
        "PDF, CSV, TXT, or image",
        type=["pdf", "csv", "txt", "png", "jpg", "jpeg", "webp"],
    )
    if uploaded_file is not None:
        kind, extracted = process_uploaded_file(uploaded_file)
        st.session_state.pending_file_context = (uploaded_file.name, kind, extracted)
        st.info(f"Loaded `{uploaded_file.name}` ({kind}). It will be added to your next message.")

    st.divider()

    # ---- CONVERSATION SEARCH ----
    st.subheader("🔍 Search Conversation")
    st.session_state.search_query = st.text_input(
        "Search messages", value=st.session_state.search_query
    )

    st.divider()

    # ---- EXPORT / IMPORT ----
    st.subheader("💾 Export / Import")

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "Markdown",
            data=export_as_markdown(st.session_state.messages),
            file_name="chat_export.md",
            mime="text/markdown",
            use_container_width=True,
        )
        st.download_button(
            "JSON",
            data=export_as_json(st.session_state.messages),
            file_name="chat_export.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_b:
        st.download_button(
            "TXT",
            data=export_as_txt(st.session_state.messages),
            file_name="chat_export.txt",
            mime="text/plain",
            use_container_width=True,
        )
        try:
            pdf_bytes = export_as_pdf(st.session_state.messages)
            st.download_button(
                "PDF",
                data=pdf_bytes,
                file_name="chat_export.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception:
            st.caption("PDF export unavailable.")

    imported = st.file_uploader("Import chat (.json)", type=["json"], key="import_uploader")
    if imported is not None:
        parsed = import_chat_from_json(imported)
        if parsed:
            st.session_state.messages = parsed
            st.success("Chat imported successfully.")
            st.rerun()
        else:
            st.error("Invalid chat file. Expecting exported JSON format.")

    st.divider()

    # ---- ANALYTICS ----
    st.subheader("📊 Session Analytics")
    analytics = compute_analytics(st.session_state.messages, st.session_state.session_start)

    m1, m2 = st.columns(2)
    m1.metric("Total Messages", analytics["total_messages"])
    m2.metric("Questions Asked", analytics["user_messages"])

    m3, m4 = st.columns(2)
    m3.metric("Assistant Replies", analytics["assistant_messages"])
    m4.metric("Est. Tokens", analytics["estimated_tokens"])

    st.metric("Session Duration", analytics["session_duration"])

    if st.session_state.last_response_time is not None:
        st.metric("Last Response Time", f"{st.session_state.last_response_time:.2f}s")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "system", "content": st.session_state.system_prompt}
        ]
        st.session_state.session_start = datetime.now()
        st.rerun()

    st.divider()
    st.markdown(
        """
        ### About
        **Deepak AI Assistant** — a professional AI companion.

        - Streamlit UI · Glassmorphism
        - Groq API · Multiple models
        - Streaming responses
        - File uploads & exports
        - Session analytics
        """
    )


# =========================
# WELCOME SCREEN (no messages yet)
# =========================

visible_messages = [m for m in st.session_state.messages if m["role"] != "system"]

if not visible_messages:
    st.markdown(
        """
        <div class="glass-card">
            <h3 style="margin-top:0;">👋 Welcome to Deepak AI</h3>
            <p style="color:#a9aec2;">
                Ask anything — code, learning, brainstorming, career advice, or just chat.
                Try one of the prompts below to get started.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    suggestions = [
        ("💻 Debug my code", "Help me debug a Python function that's throwing an error."),
        ("📚 Explain a concept", "Explain how neural networks learn, step by step."),
        ("🚀 Brainstorm ideas", "Brainstorm unique project ideas for a hackathon."),
        ("🎯 Career advice", "How do I prepare for a software engineering interview?"),
    ]

    cols = st.columns(2)
    for i, (title, prompt_text) in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(title, key=f"suggestion_{i}", use_container_width=True):
                st.session_state["_auto_prompt"] = prompt_text


# =========================
# CHAT HISTORY (with search filter)
# =========================

search_q = st.session_state.search_query.strip().lower()

for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "system":
        continue
    if search_q and search_q not in msg["content"].lower():
        continue

    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            if msg.get("timestamp"):
                st.markdown(
                    f'<div class="msg-timestamp">{msg["timestamp"]}</div>',
                    unsafe_allow_html=True,
                )
        with col2:
            st.button("📋", key=f"copy_{idx}", help="Copy response (select & Ctrl+C)")


# =========================
# CHAT INPUT
# =========================

prompt = st.chat_input("Ask me anything...")

# Allow welcome-screen suggestion buttons to trigger a prompt
if "_auto_prompt" in st.session_state and st.session_state["_auto_prompt"]:
    prompt = st.session_state.pop("_auto_prompt")

if prompt:
    # Attach any pending file context to this message
    final_prompt = prompt
    if st.session_state.pending_file_context:
        fname, kind, extracted = st.session_state.pending_file_context
        final_prompt = (
            f"{prompt}\n\n---\n"
            f"[Attached file: {fname} ({kind})]\n"
            f"{extracted}"
        )
        st.session_state.pending_file_context = None

    if not prompt.strip():
        st.error("Please enter a non-empty message.")
        st.stop()

    if not LOCAL_KEY_EXISTS and not api_key:
        st.error("Please enter your Groq API Key in the sidebar.")
        st.stop()

    timestamp = datetime.now().strftime("%H:%M:%S")

    st.session_state.messages.append(
        {"role": "user", "content": final_prompt, "timestamp": timestamp}
    )

    with st.chat_message("user", avatar="🧑"):
        st.markdown(final_prompt)
        st.markdown(f'<div class="msg-timestamp">{timestamp}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        full_response = ""
        start_time = datetime.now()

        try:
            for chunk in stream_response(
                st.session_state.messages,
                api_key=api_key,
                model=AVAILABLE_MODELS[st.session_state.model_label],
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens,
            ):
                full_response += chunk
                placeholder.markdown(full_response + " ▌")

            placeholder.markdown(full_response)
            elapsed = (datetime.now() - start_time).total_seconds()
            st.session_state.last_response_time = elapsed

            resp_timestamp = datetime.now().strftime("%H:%M:%S")
            st.markdown(
                f'<div class="msg-timestamp">{resp_timestamp} · {elapsed:.2f}s</div>',
                unsafe_allow_html=True,
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": resp_timestamp,
                }
            )

        except ChatbotError as exc:
            placeholder.error(f"⚠️ {exc}")
        except Exception as exc:  # pragma: no cover
            placeholder.error(f"⚠️ Unexpected error: {exc}")

    st.rerun()


# =========================
# FOOTER
# =========================

st.divider()
st.caption("🚀 Built by Deepak Kumar | Groq + Streamlit | Professional Edition")