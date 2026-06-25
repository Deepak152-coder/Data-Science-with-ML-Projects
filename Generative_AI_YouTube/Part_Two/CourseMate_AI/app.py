"""
CourseMate AI
=============
A Retrieval-Augmented Generation (RAG) chat application that lets users
upload a PDF and ask questions about its content.

Pipeline:
    PDF -> PyPDFLoader -> RecursiveCharacterTextSplitter -> HuggingFaceEmbeddings
        -> Chroma (vector store) -> MMR retriever -> ChatMistralAI -> Streamed answer

Run with:
    streamlit run app.py
"""

import os
import tempfile
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# ==========================================================
# SECTION 1: APP CONFIG & GLOBAL CONSTANTS
# ==========================================================

load_dotenv()

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "mistral-small-2506"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 4
RETRIEVER_FETCH_K = 10
RETRIEVER_LAMBDA_MULT = 0.5

st.set_page_config(
    page_title="CourseMate AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------
# Minimal custom CSS for a cleaner, modern, dark-theme-friendly look.
# Kept purely cosmetic — no logic lives in here.
# ----------------------------------------------------------
st.markdown(
    """
    <style>
        .stChatMessage {
            border-radius: 12px;
            padding: 0.25rem 0.5rem;
        }
        .stat-card {
            background: rgba(127, 127, 127, 0.08);
            border: 1px solid rgba(127, 127, 127, 0.2);
            border-radius: 10px;
            padding: 0.75rem 1rem;
            text-align: center;
        }
        .stat-card h3 {
            margin: 0;
            font-size: 1.4rem;
        }
        .stat-card p {
            margin: 0;
            opacity: 0.7;
            font-size: 0.8rem;
        }
        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(127, 127, 127, 0.15);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# SECTION 2: CACHED RESOURCE LOADERS
# ==========================================================
# These are expensive to (re)create, so they are cached across reruns.
# @st.cache_resource is used for objects (models, clients) that should be
# instantiated once and reused; @st.cache_data is used for pure data results.

@st.cache_resource(show_spinner=False)
def load_embedding_model() -> HuggingFaceEmbeddings:
    """Load (and cache) the HuggingFace embedding model once per session."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


@st.cache_data(show_spinner=False)
def load_and_split_pdf(file_bytes: bytes, file_name: str) -> List[Document]:
    """
    Load a PDF from raw bytes and split it into chunks.

    Cached on the file's bytes + name, so re-uploading the same file in the
    same session won't re-parse or re-split it.
    """
    # PyPDFLoader needs a real file path, so we write to a temp file.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(file_bytes)
        pdf_path = temp_pdf.name

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(documents)

        # Stash page count on the chunks list via a side-channel return
        # (handled by caller using len(documents)) — kept simple here.
        return chunks
    finally:
        os.remove(pdf_path)


def build_vectorstore(chunks: List[Document]) -> Chroma:
    """
    Build a fresh, in-memory Chroma vector store from document chunks.

    Not cached with @st.cache_resource because each new PDF needs its own
    store; caching here would silently reuse a stale store across documents.
    Embeddings themselves are still fast because the embedding MODEL is cached.
    """
    embedding_model = load_embedding_model()
    return Chroma.from_documents(documents=chunks, embedding=embedding_model)


# ==========================================================
# SECTION 3: PROMPT TEMPLATE
# ==========================================================
# Tightened to explicitly forbid outside knowledge and guesswork,
# which reduces hallucination compared to a loosely worded system prompt.

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are CourseMate AI, a precise document-answering assistant.

Rules you must follow strictly:
1. Answer ONLY using the information in the provided context below.
2. Do NOT use any outside knowledge, assumptions, or guesses.
3. If the context does not contain enough information to answer confidently,
   reply EXACTLY with: "I could not find the answer in the document."
4. Do not fabricate page numbers, facts, or details not present in the context.
5. Keep answers concise and directly relevant to the question asked.""",
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}""",
        ),
    ]
)


# ==========================================================
# SECTION 4: SESSION STATE INITIALIZATION
# ==========================================================

def init_session_state() -> None:
    """Initialize all session state keys used throughout the app, once."""
    defaults = {
        "retriever": None,
        "messages": [],
        "vectorstore": None,
        "processed_file_name": None,
        "total_pages": 0,
        "total_chunks": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ==========================================================
# SECTION 5: SIDEBAR — API KEY, PDF UPLOAD, DOCUMENT STATS
# ==========================================================

with st.sidebar:
    st.markdown("## 📚 CourseMate AI")
    st.caption("RAG-powered PDF Q&A — HuggingFace + Chroma + Mistral")
    st.divider()

    # ---- API key handling ----
    st.markdown("### 🔑 Mistral API Key")
    env_api_key = os.getenv("MISTRAL_API_KEY")

    if env_api_key:
        st.success("Using API key from .env", icon="✅")
        api_key: Optional[str] = env_api_key
    else:
        api_key = st.text_input(
            "Enter your Mistral API Key",
            type="password",
            help="No .env key found — please provide your own key.",
        )
        if not api_key:
            st.warning("API key required to chat.", icon="⚠️")

    st.divider()

    # ---- PDF upload ----
    st.markdown("### 📄 Document")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    # "Process New PDF" lets the user reset and load a different document
    # without restarting the whole app.
    if st.session_state.processed_file_name is not None:
        if st.button("🔄 Process New PDF", use_container_width=True):
            st.session_state.retriever = None
            st.session_state.vectorstore = None
            st.session_state.processed_file_name = None
            st.session_state.total_pages = 0
            st.session_state.total_chunks = 0
            st.session_state.messages = []
            st.rerun()

    # ---- Clear chat ----
    if st.session_state.messages:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ---- Document stats ----
    if st.session_state.processed_file_name:
        st.divider()
        st.markdown("### 📊 Document Stats")
        st.markdown(f"**File:** {st.session_state.processed_file_name}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""<div class="stat-card"><h3>{st.session_state.total_pages}</h3>
                <p>Pages</p></div>""",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""<div class="stat-card"><h3>{st.session_state.total_chunks}</h3>
                <p>Chunks</p></div>""",
                unsafe_allow_html=True,
            )


# ==========================================================
# SECTION 6: PDF PROCESSING PIPELINE
# ==========================================================

def process_uploaded_pdf(uploaded_file) -> None:
    """
    Run the full ingestion pipeline for a newly uploaded PDF:
    load -> split -> embed -> store -> build retriever.

    Wrapped in a try/except so a corrupt PDF or embedding failure shows a
    clean error instead of crashing the app.
    """
    try:
        with st.status("Processing PDF...", expanded=True) as status:
            file_bytes = uploaded_file.read()

            status.write("📖 Loading and splitting PDF into chunks...")
            chunks = load_and_split_pdf(file_bytes, uploaded_file.name)

            if not chunks:
                status.update(label="No extractable text found.", state="error")
                st.error(
                    "No text could be extracted from this PDF. "
                    "It may be a scanned/image-only document."
                )
                return

            # Recompute page count separately (cheap) for display purposes.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                page_count = len(PyPDFLoader(tmp_path).load())
            finally:
                os.remove(tmp_path)

            status.write("🧠 Generating embeddings (HuggingFace)...")
            vectorstore = build_vectorstore(chunks)

            status.write("🗂️ Building retriever (MMR search)...")
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": RETRIEVER_K,
                    "fetch_k": RETRIEVER_FETCH_K,
                    "lambda_mult": RETRIEVER_LAMBDA_MULT,
                },
            )

            # Persist results in session state
            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = retriever
            st.session_state.processed_file_name = uploaded_file.name
            st.session_state.total_pages = page_count
            st.session_state.total_chunks = len(chunks)

            status.update(label="✅ PDF processed successfully!", state="complete")

    except Exception as exc:  # noqa: BLE001 - surfaced to the user deliberately
        st.error(f"❌ Failed to process PDF: {exc}")


# Trigger processing only when a *new* file is uploaded (different from the
# one already processed), avoiding redundant reprocessing on every rerun.
if uploaded_pdf and uploaded_pdf.name != st.session_state.processed_file_name:
    process_uploaded_pdf(uploaded_pdf)


# ==========================================================
# SECTION 7: MAIN CHAT INTERFACE
# ==========================================================

st.title("📚 CourseMate AI")
st.caption("Upload a PDF on the left, then ask questions about it below.")

# ---- Render existing chat history ----
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Re-render saved source chunks for past assistant messages, if any.
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📎 View retrieved sources"):
                for i, src in enumerate(message["sources"], start=1):
                    score_text = (
                        f" — similarity score: {src['score']:.4f}"
                        if src.get("score") is not None
                        else ""
                    )
                    st.markdown(f"**Chunk {i}** (page {src['page']}){score_text}")
                    st.text(src["content"])

# ---- Chat input ----
query = st.chat_input("Ask a question about your PDF...")

if query:
    # ---- Guard clauses: fail fast with clear messages ----
    if not api_key:
        st.error("Please enter your Mistral API key in the sidebar.")
        st.stop()

    if st.session_state.retriever is None:
        st.error("Please upload and process a PDF first.")
        st.stop()

    # Show the user's message immediately
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("🔍 Retrieving relevant sections..."):
                # Use similarity_search_with_score directly on the vectorstore
                # so we can surface scores; MMR retriever itself doesn't
                # expose scores, so we fetch with scores for display while
                # still using the MMR-selected docs for context generation.
                retrieved_docs: List[Document] = st.session_state.retriever.invoke(
                    query
                )

                # Best-effort similarity scores for display purposes only.
                scored_docs = []
                try:
                    scored_docs = st.session_state.vectorstore.similarity_search_with_score(
                        query, k=RETRIEVER_K
                    )
                except Exception:
                    scored_docs = []

            # Prevent an unnecessary, low-value LLM call when retrieval
            # returns nothing at all (e.g. empty/irrelevant document).
            if not retrieved_docs:
                answer = "I could not find the answer in the document."
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": []}
                )
                st.stop()

            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

            final_prompt = RAG_PROMPT.invoke(
                {"context": context_text, "question": query}
            )

            llm = ChatMistralAI(
                model=LLM_MODEL_NAME,
                api_key=api_key,
                streaming=True,
            )

            # ---- Stream the response token-by-token ----
            def response_generator():
                for chunk in llm.stream(final_prompt):
                    if chunk.content:
                        yield chunk.content

            answer = st.write_stream(response_generator())

            # ---- Build source metadata for the expander ----
            score_lookup = {id(doc): score for doc, score in scored_docs} if scored_docs else {}
            sources = []
            for doc in retrieved_docs:
                # scored_docs may contain different Document instances than
                # retrieved_docs (separate calls), so fall back gracefully.
                score = None
                for sdoc, sscore in scored_docs:
                    if sdoc.page_content == doc.page_content:
                        score = sscore
                        break
                sources.append(
                    {
                        "content": doc.page_content,
                        "page": doc.metadata.get("page", "N/A"),
                        "score": score,
                    }
                )

            if sources:
                with st.expander("📎 View retrieved sources"):
                    for i, src in enumerate(sources, start=1):
                        score_text = (
                            f" — similarity score: {src['score']:.4f}"
                            if src["score"] is not None
                            else ""
                        )
                        st.markdown(f"**Chunk {i}** (page {src['page']}){score_text}")
                        st.text(src["content"])

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )

        except Exception as exc:  # noqa: BLE001 - surfaced to the user deliberately
            error_message = f"⚠️ Something went wrong while generating a response: {exc}"
            st.error(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message, "sources": []}
            )