import streamlit as st
import os
from dotenv import load_dotenv
from chatbot import get_response

load_dotenv()

LOCAL_KEY_EXISTS = bool(
    os.getenv("GROQ_API_KEY")
)

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Deepak AI",
    page_icon="🤖",
    layout="wide"
)

# =========================
# SYSTEM PROMPT
# =========================

SYSTEM_PROMPT = """
You are Deepak AI Assistant, a friendly, intelligent, and supportive AI companion.

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

You are not just an information provider; you are a helpful companion who assists with learning, problem-solving, projects, career growth, and everyday questions.
"""

# =========================
# SESSION STATE
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.chat-title {
    text-align: center;
    font-size: 3rem;
    font-weight: bold;
    background: linear-gradient(90deg,#00DBDE,#FC00FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 20px;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================

st.markdown(
    '<p class="chat-title">🤖 Deepak AI Assistant</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Powered by Groq + Llama 3.3 70B</p>',
    unsafe_allow_html=True
)

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.title("⚙️ Control Panel")

    st.info("🚀 Groq Llama 3.3 70B")

    if not LOCAL_KEY_EXISTS:

        api_key = st.text_input(
            "🔑 Enter Groq API Key",
            type="password",
            help="Required when running on Streamlit Cloud."
        )

        st.caption(
            "Get your key from https://console.groq.com/keys"
        )

    else:

        api_key = None

        st.success(
            "✅ Using local .env API key"
        )

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        st.rerun()

    st.divider()

    visible_messages = [
        msg
        for msg in st.session_state.messages
        if msg["role"] != "system"
    ]

    total_messages = len(
        visible_messages
    )

    user_messages = len([
        msg
        for msg in visible_messages
        if msg["role"] == "user"
    ])

    st.metric(
        "Total Messages",
        total_messages
    )

    st.metric(
        "Questions Asked",
        user_messages
    )

    st.divider()

    st.markdown("""
    ### About

    Custom AI Chatbot

    - Streamlit UI
    - Groq API
    - Llama 3.3 70B
    - Chat History Support
    - Friendly AI Companion
    """)

# =========================
# CHAT HISTORY
# =========================

for msg in st.session_state.messages:

    if msg["role"] == "system":
        continue

    with st.chat_message(msg["role"]):
        st.markdown(
            msg["content"]
        )

# =========================
# CHAT INPUT
# =========================

prompt = st.chat_input(
    "Ask me anything..."
)

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    if not LOCAL_KEY_EXISTS and not api_key:

        st.error(
            "Please enter your Groq API Key in the sidebar."
        )

        st.stop()

    with st.spinner(
        "🤔 Thinking..."
    ):

        response = get_response(
            st.session_state.messages,
            api_key
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    st.rerun()

# =========================
# FOOTER
# =========================

st.divider()

st.caption(
    "🚀 Built by Deepak Kumar | Groq + Streamlit"
)