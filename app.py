import os

# Set before importing Streamlit.
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import hashlib
import threading

import streamlit as st

from create_database import (
    load_embeddings,
    open_or_create_database,
)
from main import (
    create_llm,
    generate_answer,
    retrieve_documents,
)

st.set_page_config(
    page_title="Book Buddy",
    page_icon="📚",
    layout="centered",
)

st.title("📚 Book Buddy")
st.caption("Upload a PDF and ask questions about its contents.")

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY missing. Check your project's .env file.")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_embedding_resources():
    # Shared model; serialize access across browser sessions.
    return load_embeddings(), threading.Lock()


@st.cache_resource(show_spinner=False)
def get_llm():
    return create_llm()


if "book_store" not in st.session_state:
    st.session_state.book_store = None
    st.session_state.book_hash = None
    st.session_state.chat = []

try:
    with st.spinner("Loading embedding model..."):
        embeddings, embedding_lock = get_embedding_resources()
except Exception as error:
    st.error(f"Could not load embedding model: {error}")
    st.stop()

with st.sidebar:
    st.header("Your Book")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
    )

if uploaded_file is None:
    st.info("Upload a PDF to begin.")
    st.stop()

pdf_bytes = uploaded_file.getvalue()
selected_hash = hashlib.sha256(pdf_bytes).hexdigest()

# Automatically process only when the selected book changes.
if (
    st.session_state.book_store is None
    or st.session_state.book_hash != selected_hash
):
    status = st.empty()

    try:
        status.info("Preparing your book...")

        with embedding_lock:
            store = open_or_create_database(
                pdf_bytes,
                embeddings,
                progress=status.info,
            )

        st.session_state.book_store = store
        st.session_state.book_hash = selected_hash
        st.session_state.chat = []

        status.empty()

    except Exception as error:
        status.empty()
        st.error(f"Could not process PDF: {error}")
        st.stop()

st.caption(f"Ready: {uploaded_file.name}")

for message in st.session_state.chat:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your book...")

if question and question.strip():
    question = question.strip()

    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.spinner("Searching the book..."):
            with embedding_lock:
                documents = retrieve_documents(
                    st.session_state.book_store,
                    question,
                )

        with st.spinner("Waiting for Groq..."):
            answer = generate_answer(
                get_llm(),
                documents,
                question,
            )

        st.session_state.chat.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])

        with st.chat_message("assistant"):
            st.markdown(answer)

    except Exception as error:
        if "429" in str(error):
            st.error("Groq rate limit reached. Please try again later.")
        else:
            st.error(f"Request failed: {error}")