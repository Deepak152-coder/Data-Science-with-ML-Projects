import streamlit as st
from dotenv import load_dotenv
import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI

load_dotenv()

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Mistral Chatbot",
    page_icon="🤖"
)

st.title("🤖 Mistral Chatbot")

# --------------------------------------------------
# Personalities
# --------------------------------------------------
PERSONALITIES = {
    "Helpful": "You are a helpful AI assistant.",
    "Funny": "You are a funny AI assistant who loves jokes and humor.",
    "Teacher": "You are an expert teacher who explains concepts step-by-step.",
    "Motivator": "You are a motivational coach who inspires users.",
    "Professional": "You are a professional AI assistant. Be concise and formal.",
    "Pirate": "You are a pirate. Speak like a pirate.",
    "Angry": "You are an angry AI assistant. Respond aggressively but without abusive language.",
    "Interviewer": "You are a technical interviewer. Ask follow-up questions and evaluate answers.",
    "Coding Mentor": "You are an expert coding mentor helping users learn programming.",
    "Data Scientist": "You are a senior data scientist helping with machine learning, AI, and analytics.",
    "DSA Coach": "You are a DSA coach helping users learn algorithms and crack coding interviews."
}

# --------------------------------------------------
# Model
# --------------------------------------------------
@st.cache_resource
def get_model():
    return ChatMistralAI(
        model_name="mistral-small-2506",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.7,
    )

model = get_model()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    selected_personality = st.selectbox(
        "Choose Personality",
        list(PERSONALITIES.keys())
    )

    if (
        "current_personality" not in st.session_state
        or st.session_state.current_personality != selected_personality
    ):
        st.session_state.current_personality = selected_personality

        st.session_state.messages = [
            SystemMessage(
                content=PERSONALITIES[selected_personality]
            )
        ]

    st.markdown("---")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [
            SystemMessage(
                content=PERSONALITIES[selected_personality]
            )
        ]
        st.rerun()

# --------------------------------------------------
# Session State
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(
            content=PERSONALITIES[selected_personality]
        )
    ]

# --------------------------------------------------
# Display Chat History
# --------------------------------------------------
for msg in st.session_state.messages:

    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)

    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# --------------------------------------------------
# Chat Input
# --------------------------------------------------
prompt = st.chat_input("Type your message...")

if prompt:

    st.session_state.messages.append(
        HumanMessage(content=prompt)
    )

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:
                response = model.invoke(
                    st.session_state.messages
                )

                st.write(response.content)

                st.session_state.messages.append(
                    AIMessage(content=response.content)
                )

            except Exception as e:
                st.error(f"Error: {e}")