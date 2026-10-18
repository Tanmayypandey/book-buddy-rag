# 📚 Book Buddy — PDF Question Answering with RAG

Book Buddy helps users explore PDF books through natural-language questions. Upload a text-based PDF, ask a question, and receive an answer generated from relevant passages in the document.

Built with Python, Streamlit, LangChain, Chroma, Hugging Face embeddings, and Groq.

## Features

- Upload a PDF through a Streamlit interface.
- Automatically extract text and split it into overlapping chunks.
- Generate embeddings locally without a paid embedding API.
- Store and reuse document indexes for identical PDFs.
- Retrieve relevant passages using Maximum Marginal Relevance (MMR).
- Generate answers using Groq’s hosted `openai/gpt-oss-20b` model.
- Display previous questions and answers during the browser session.
- Handle missing text, API failures, and rate-limit errors.

## How It Works

1. The user uploads a PDF.
2. PyPDFLoader extracts text from its pages.
3. RecursiveCharacterTextSplitter divides the text into chunks.
4. `all-MiniLM-L6-v2` converts chunks into embeddings.
5. Chroma stores the vectors.
6. MMR retrieval selects relevant passages for each question.
7. Groq generates an answer using the retrieved context.

The prompt instructs the model to acknowledge when the document does not contain the answer. Generated answers should still be checked against the source.

## Technology Stack

| Component | Technology |
|---|---|
| Interface | Streamlit |
| RAG orchestration | LangChain |
| PDF extraction | PyPDFLoader / pypdf |
| Text splitting | RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector database | Chroma |
| Retrieval | MMR |
| Answer generation | Groq — openai/gpt-oss-20b |
| Environment configuration | python-dotenv |

## Project Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface and upload handling |
| `main.py` | Retrieval and answer-generation functions |
| `create_database.py` | Embedding model loading and PDF indexing |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Streamlit configuration |
| `.gitignore` | Excludes secrets and generated files |

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/Tanmayypandey/book-buddy-rag.git
cd book-buddy-rag
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the Groq key

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY
```

Never commit `.env` or API keys to GitHub.

### 5. Start the application

```bash
python -m streamlit run app.py --server.fileWatcherType none
```

Open the local URL displayed in the terminal.

## Usage

1. Upload a text-based PDF.
2. Wait for the book to finish processing.
3. Enter a question about its contents.
4. Read the generated answer.
5. Upload another PDF to ask questions about a different book.

Example questions:

- What is the main topic of this book?
- Explain gradient descent according to the document.
- How does the document describe recurrent neural networks?

## Performance and Storage

- The embedding model is cached within the running server.
- Identical PDFs reuse a saved index when it remains available.
- New PDFs require initial embedding and indexing.
- Processing time depends on document size, hardware, and server load.
- Indexes are saved locally in `book_databases/`.
- Local indexes may be lost when hosting uses an ephemeral filesystem.

## Limitations

- Scanned or image-only PDFs require OCR, which is not included.
- Complex tables, formulas, and layouts may extract imperfectly.
- Retrieved passages may not contain every relevant detail.
- Displayed chat history is not passed as conversational memory to the model.
- Groq API rate limits apply; deployed users share the configured API quota.
- The current app has no authentication or user-specific persistent storage.
- Relevant document passages are sent to Groq for answer generation.

## Deployment Status

The application works locally. A Render deployment has been created, but the tested 512 MB instance runs out of memory while loading the embedding stack.

Hosted availability and performance are still being evaluated.

## What I Learned

- Building a complete retrieval-augmented generation workflow
- Working with PDF loaders and overlapping text chunks
- Creating and persisting vector embeddings
- Using MMR retrieval with Chroma
- Integrating Groq with LangChain
- Managing Streamlit sessions and cached resources
- Debugging API limits and cloud memory constraints

## Author

**Tanmay Pandey**

- GitHub: https://github.com/Tanmayypandey
- LinkedIn: https://www.linkedin.com/in/tanmay-pandey-9b37bb278
