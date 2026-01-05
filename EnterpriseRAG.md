# Enterprise RAG

Enterprise RAG is a small end‑to‑end Retrieval‑Augmented Generation (RAG) system:

- **Backend**: FastAPI service that ingests PDFs, chunks and embeds them, stores vectors in Qdrant, and exposes semantic search + RAG “ask” endpoints.
- **Vector store**: Qdrant (local Docker or Qdrant Cloud).
- **LLM**: Google Gemini (via `google-genai` / LangChain).
- **UI**: Streamlit app (`ui.py`) that lets non‑technical users upload PDFs, ask grounded questions, and delete documents.

The project is designed for:

- Demos and interviews
- Leads and stakeholders to see “how it works” in a **Leads / Demo** tab
- End‑users to just upload PDFs and ask questions in a **User** tab

## 1. Architecture

### 1.1 Backend (FastAPI)

Entry point: [main.py](file:///d:/enterprise-rag/main.py)

- **Routers**
  - `/health` – basic and readiness health checks
  - `/ingest` – document ingestion and deletion
  - `/query/search` – semantic search over chunks
  - `/query/ask` – full RAG pipeline: retrieve + generate answer + citations
- **Core components** (under [app/core](file:///d:/enterprise-rag/app)):
  - `document_processor.py`
    - Extracts text from PDF with `pypdf`
    - Cleans and splits into overlapping chunks (configurable size/overlap)
    - Generates `document_id` and `chunk_id`s
  - `embeddings.py`
    - SentenceTransformers model (`all-MiniLM-L6-v2` by default)
  - `vector_store.py`
    - Qdrant client wrapper
    - `upsert_chunks`: stores chunk embeddings and payloads
    - `delete_document`: removes all points with a given `document_id`
  - `retriever.py`
    - Encodes queries and searches Qdrant
    - Supports:
      - `document_id` (single‑document filter)
      - `document_ids` (list of docs; “files at hand” mode)
  - `generator.py`
    - Calls Gemini via LangChain with retrieved context to generate grounded answers
- **Models** (under [app/models/schemas.py](file:///d:/enterprise-rag/app/models/schemas.py)):
  - `IngestResponse`, `SearchRequest/Response`, `AskRequest/Response`, `SourceCitation`, etc.

Global configuration is centralized in [app/config.py](file:///d:/enterprise-rag/app/config.py) via `pydantic-settings`.

### 1.2 Vector Store (Qdrant)

- Configured via environment variables:
  - `QDRANT_URL` / `QDRANT_API_KEY` for Qdrant Cloud
  - Or `QDRANT_HOST` / `QDRANT_PORT` for local Docker Qdrant
- Collections:
  - Default collection name: `documents`
  - Payload includes:
    - `document_id`
    - `chunk_id`
    - `text`
    - `page_number`
    - `chunk_index`
    - additional metadata

The backend validates Qdrant connectivity on startup (`create_app()` lifespan handler).

### 1.3 LLM (Gemini)

- Uses `google-genai` + LangChain (`langchain-google-genai`).
- Config:
  - `GOOGLE_API_KEY` (in environment) must be set on the backend.
- RAG pipeline:
  - Retrieve relevant chunks from Qdrant
  - Build a prompt with those chunks
  - Call Gemini to answer, returning:
    - `answer`
    - list of `SourceCitation` with `document_id`, `page_number`, `text_preview`, `relevance_score`

### 1.4 UI (Streamlit)

Single‑page app: [ui.py](file:///d:/enterprise-rag/ui.py)

- Talks to the FastAPI backend via HTTP requests.
- Reads configuration from:
  - Streamlit secrets (`st.secrets`) – recommended for Streamlit Cloud
  - Or environment variables
- Main features:
  - Password‑protected access (`UI_PASSWORD`)
  - **Two tabs**:
    - **Leads / Demo**
      - Shows health, advanced ingestion/search/ask pages, and technical details for interviews/demos.
    - **User**
      - Simplified experience:
        - Upload/Ingest PDFs
        - Ask questions
        - Delete documents
      - Supports NotebookLM‑style **“Files at hand” vs “General knowledge”** scoping.

## 2. Features

### 2.1 Document Ingestion

Endpoint: `POST /ingest`

Flow:

1. Validate file (PDF only, size limit from config).
2. Extract text from PDF pages.
3. Chunk text with overlap and sentence‑aware boundaries.
4. Generate embeddings with SentenceTransformers.
5. Upsert into Qdrant with payload metadata.
6. Return:
   - `document_id`
   - filename
   - number of chunks
   - number of pages

The UI:

- In the **User** tab:
  - Users can upload a PDF and see success state.
  - Internally the UI remembers each ingested document in session (`st.session_state.documents`).
- In the **Leads / Demo** tab:
  - There is a more detailed ingest page showing last responses and technical metadata (optional).

### 2.2 Semantic Search

Endpoints:

- `POST /query/search` (primary)
- `GET /query/search` (convenience)

Options:

- `query` – natural language query
- `top_k` – number of results
- `score_threshold` – filter by similarity
- `document_id` – restrict to a single document
- `document_ids` – restrict to a list of documents (used by “files at hand” mode)

The UI shows:

- Result table (chunk_id, document_id, page, score, preview)
- Altair bar chart of scores
- Expanders with full chunk text and optional metadata

### 2.3 Ask (RAG)

Endpoint: `POST /query/ask`

Request:

- `question`
- `top_k`
- `temperature`
- `document_id` (optional)
- `document_ids` (optional; list of docs → “files at hand” mode)

Response:

- `answer` (Gemini)
- `sources` – list of citations with source_id, document_id, page_number, text_preview, relevance_score
- `model` – model name used

The UI:

- Display answer and collapsible list of sources.
- In the **User** tab, users can choose:

  - **Knowledge base: Files at hand**
    - Multi‑select from PDFs ingested in the current session.
    - The UI sends `document_ids` to `/query/ask`, so answers are limited to those docs.
  - **Knowledge base: General knowledge**
    - Sends no `document_ids`; the backend retrieves from the entire vector store.

### 2.4 Document Deletion

Endpoint: `DELETE /ingest/{document_id}`

- Deletes all Qdrant points for a given `document_id`.

The UI:

- **User** tab:
  - A “Delete a document” section where the user chooses from known documents (session history).
  - No need to type `document_id` manually.
- **Leads / Demo** tab:
  - Advanced delete controls with access to raw `document_id` if needed.

## 3. Configuration

### 3.1 Backend (FastAPI)

Key environment variables (see [app/config.py](file:///d:/enterprise-rag/app/config.py)):

- General:
  - `APP_NAME` (optional, defaults to `Enterprise RAG`)
  - `APP_VERSION` (optional)
  - `DEBUG` (default: `False`)
- Security:
  - `API_KEY` (optional API key for backend)
  - `API_KEY_HEADER` (default: `X-API-Key`)
  - `CORS_ORIGINS` (comma‑separated or `*`)
- Qdrant:
  - `QDRANT_URL` (for cloud; if set, used with `QDRANT_API_KEY`)
  - `QDRANT_API_KEY` (for cloud)
  - `QDRANT_HOST` (default `qdrant`)
  - `QDRANT_PORT` (default `6333`)
  - `QDRANT_COLLECTION_NAME` (default `documents`)
- Gemini:
  - `GOOGLE_API_KEY` (required for RAG `/query/ask`)
- Embeddings:
  - `EMBEDDING_MODEL`
  - `EMBEDDING_DIMENSION`
- Chunking:
  - `CHUNK_SIZE`
  - `CHUNK_OVERLAP`

### 3.2 UI (Streamlit)

The UI reads configuration from:

- Streamlit secrets (`st.secrets`) – preferred on Streamlit Cloud.
- Environment variables as fallback.

Important keys:

- `RAG_API_URL` – base URL of the FastAPI backend.
  - Example: `http://localhost:8000` (local)
  - Example (cloud): `https://your-api-service.onrender.com`
- `RAG_API_KEY_HEADER` – header name for API key (matches backend `API_KEY_HEADER`).
- `RAG_API_KEY` – API key value, if backend security is enabled.
- `RAG_VERIFY_TLS` – `"true"` / `"false"` to enable/disable TLS verification.
- `RAG_TIMEOUT_S` – request timeout in seconds.
- `UI_PASSWORD` – optional password for the UI itself.

The UI also normalizes `RAG_API_URL` to ensure:

- There is a scheme (`http://` added if missing).
- No double slashes at the end.

## 4. Running Locally

### 4.1 Prerequisites

- Python 3.12+
- Qdrant:
  - Either run in Docker, or use Qdrant Cloud and set `QDRANT_URL` + `QDRANT_API_KEY`.
- A Gemini API key (`GOOGLE_API_KEY`).

Install dependencies:

```bash
pip install -r requirements.txt
```

or via `uv` / `pip` using `pyproject.toml`.

### 4.2 Start Qdrant (local example)

Minimal Docker example:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Set env vars to point to local Qdrant if needed:

```bash
set QDRANT_HOST=localhost
set QDRANT_PORT=6333
```

### 4.3 Start the FastAPI backend

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or just run:

```bash
python main.py
```

### 4.4 Start the Streamlit UI

Set `RAG_API_URL` so the UI knows where the backend is:

```bash
set RAG_API_URL=http://localhost:8000
```

Then:

```bash
streamlit run ui.py
```

In your browser, you will see:

- A password prompt (if `UI_PASSWORD` is set).
- Tabs:
  - **Leads / Demo**
  - **User**

## 5. Deployment Notes

### 5.1 Backend

You can deploy the FastAPI backend to any platform that supports ASGI:

- Render, Fly.io, Azure App Service, etc.
- Use `uvicorn` with the application object `app` from `main.py`.

For production:

- Set `DEBUG=false`.
- Configure proper `CORS_ORIGINS`.
- Secure with `API_KEY` (and TLS).

### 5.2 UI (Streamlit Cloud)

For Streamlit Cloud:

1. Point the app to `ui.py`.
2. In Streamlit **Secrets**, define:
   - `RAG_API_URL`: URL of your deployed FastAPI backend.
   - `RAG_API_KEY_HEADER` / `RAG_API_KEY` if you use backend API key auth.
   - `UI_PASSWORD` if you want password protection.
3. The UI will automatically:
   - Normalize the API base URL.
   - Use secure backend connection.
   - Hide Streamlit default menu/footer for a cleaner look.

## 6. Development & Quality

### 6.1 Pre-commit hooks

The repo is configured with pre‑commit:

- `ruff` / `ruff-format`
- `mypy`
- basic hygiene checks (EoF, trailing whitespace, YAML/TOML validation, etc.)

To run the same checks manually:

```bash
bash check.sh
```

or directly:

```bash
ruff check .
ruff format .
mypy app ui.py
```

### 6.2 Code Style

- Type annotations (Python 3.12, Pydantic v2).
- FastAPI routers grouped by domain (`health`, `ingest`, `query`).
- Vector store wrapped behind `VectorStore` for easier swapping.
- UI avoids exposing internal details to end‑users by default, but the **Leads / Demo** tab keeps everything accessible for debugging and explanation.

## 7. Typical Flows

### 7.1 NotebookLM‑style “Files at hand” flow

1. Go to **User** tab.
2. Upload one or more PDFs (Ingest section).
3. In the Ask section:
   - Choose **Knowledge base: Files at hand**.
   - Optionally select a subset of uploaded PDFs.
4. Ask a question.
5. Backend receives `document_ids` for those PDFs and restricts retrieval to them.

### 7.2 General knowledge flow

1. Ingest many PDFs over time.
2. In the UI, choose **Knowledge base: General knowledge**.
3. Ask questions without scoping.
4. Backend retrieves from the full collection in Qdrant.

---

This README is focused on the current code layout (`app/` + `ui.py`). If you change the project structure or add new services, update the relevant sections accordingly.
