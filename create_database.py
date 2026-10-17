import hashlib
import tempfile
from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parent
DATABASE_ROOT = PROJECT_PATH / "book_databases"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_VERSION = "minilm-1000-150-v2"


def load_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 32},
    )


def open_or_create_database(pdf_bytes, embeddings, progress=print):
    from filelock import FileLock
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    fingerprint = hashlib.sha256(
        pdf_bytes + f"{MODEL_NAME}|{INDEX_VERSION}".encode()
    ).hexdigest()

    DATABASE_ROOT.mkdir(parents=True, exist_ok=True)

    db_path = DATABASE_ROOT / fingerprint
    ready_file = db_path / "READY"

    progress("Checking saved database...")

    with FileLock(
        str(DATABASE_ROOT / f"{fingerprint}.lock"),
        timeout=300,
    ):
        if ready_file.is_file():
            progress("Opening saved database...")

            return Chroma(
                collection_name="book",
                persist_directory=str(db_path),
                embedding_function=embeddings,
            )

        progress("Reading PDF...")

        with tempfile.TemporaryDirectory() as temporary_directory:
            pdf_path = Path(temporary_directory) / "book.pdf"
            pdf_path.write_bytes(pdf_bytes)

            documents = PyPDFLoader(str(pdf_path)).load()

        documents = [
            doc for doc in documents
            if doc.page_content.strip()
        ]

        if not documents:
            raise ValueError(
                "No readable text found. This PDF may require OCR."
            )

        for doc in documents:
            doc.metadata["source"] = "uploaded_book"

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )

        chunks = splitter.split_documents(documents)

        progress(f"Preparing {len(chunks)} text chunks...")

        store = Chroma(
            collection_name="book",
            persist_directory=str(db_path),
            embedding_function=embeddings,
        )

        # Remove an incomplete collection from an interrupted run.
        store.delete_collection()

        store = Chroma(
            collection_name="book",
            persist_directory=str(db_path),
            embedding_function=embeddings,
        )

        batch_size = 32

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]

            store.add_documents(
                documents=batch,
                ids=[
                    f"chunk_{index}"
                    for index in range(start, start + len(batch))
                ],
            )

            progress(
                f"Processing: {start + len(batch)}/{len(chunks)} chunks"
            )

        # Mark complete only after all chunks have been saved.
        ready_file.write_text("complete", encoding="utf-8")

        progress("Book ready.")
        return store


if __name__ == "__main__":
    pdf_path = PROJECT_PATH / "document loader" / "deeplearning.pdf"

    if not pdf_path.is_file():
        raise SystemExit(f"PDF not found: {pdf_path}")

    print("Loading embedding model...", flush=True)
    embedding_model = load_embeddings()

    open_or_create_database(
        pdf_path.read_bytes(),
        embedding_model,
        progress=lambda text: print(text, flush=True),
    )