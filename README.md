# 📚 Enterprise RAG Platform

A production-ready Retrieval-Augmented Generation (RAG) system for intelligent document Q&A. Upload PDFs, ask questions, and get AI-powered answers with source citations and highlighting.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green) ![License](https://img.shields.io/badge/License-MIT-yellow) ![Docker](https://img.shields.io/badge/Docker-Ready-blue)

## ✨ Features

* **📄 PDF Ingestion** - Upload and process PDF documents with automatic text extraction
* **🔍 Semantic Search** - Find relevant content using state-of-the-art embeddings
* **💬 AI-Powered Q&A** - Get accurate answers grounded in your documents
* **📍 Source Highlighting** - See exactly where answers come from with character-level highlighting
* **📑 Multi-Document Filtering** - Query specific documents or your entire knowledge base
* **🎨 Modern UI** - Clean, responsive interface inspired by NotebookLM
* **🔌 Multi-LLM Support** - Works with Groq, OpenAI, Anthropic, and Google AI
* **🚀 Production Ready** - Docker support, rate limiting, API key auth, comprehensive logging

## 🖥️ Demo

Deploy your own instance on Hugging Face Spaces (free):

[![Try A Demo](https://img.shields.io/badge/Try%20A%20Demo-Hugging%20Face-yellow?logo=huggingface)](https://huggingface.co/spaces/A-Elshahawy/enterprise_rag)

## 📋 Table of Contents

* [Quick Start](https://claude.ai/chat/86ff3a13-df1a-4041-a1ea-7bd8069674a4#-quick-start)
* [Installation](https://claude.ai/chat/86ff3a13-df1a-4041-a1ea-7bd8069674a4#-installation)
* [Configuration](https://claude.ai/chat/86ff3a13-df1a-4041-a1ea-7bd8069674a4#-configuration)
* [Usage](https://claude.ai/chat/86ff3a13-df1a-4041-a1ea-7bd8069674a4#-usage)
* [API Reference](https://claude.ai/chat/86ff3a13-df1a-4041-a1ea-7bd8069674a4#-api-reference)
* [Architecture](https://claude.ai/chat/86ff3a13-df1a-4041-a1ea-7bd8069674a4#-architecture)
* [Deployment](https://claude.ai/chat/86ff3a13-df1a-4041-a1ea-7bd8069674a4#-deployment)
* [Contributing](https://claude.ai/chat/86ff3a13-df1a-4041-a1ea-7bd8069674a4#-contributing)

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/enterprise-rag.git
cd enterprise-rag

# Set your API key
export GROQ_API_KEY=your_groq_api_key

# Run with Docker
docker build -t enterprise-rag .
docker run -p 7860:7860 -e GROQ_API_KEY=$GROQ_API_KEY enterprise-rag
```

Open http://localhost:7860 in your browser.

### Using Python

```bash
# Clone and install
git clone https://github.com/yourusername/enterprise-rag.git
cd enterprise-rag
pip install -r requirements.txt

# Set environment variable
export GROQ_API_KEY=your_groq_api_key

# Run
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## 📦 Installation

### Prerequisites

* Python 3.10+
* 4GB RAM minimum (for embedding model)
* API key from at least one LLM provider

### Local Development Setup

```bash
# 1. Clone repository
git clone https://github.com/yourusername/enterprise-rag.git
cd enterprise-rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file
cp .env.example .env
# Edit .env with your API keys

# 5. Run the application
uvicorn main:app --reload --port 8000
```

### Using Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  rag:
    build: .
    ports:
      - "7860:7860"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - QDRANT_IN_MEMORY=true
    volumes:
      - ./data:/app/data
```

```bash
docker-compose up -d
```

## ⚙️ Configuration

### Environment Variables

| Variable              | Required | Default              | Description                                                 |
| --------------------- | -------- | -------------------- | ----------------------------------------------------------- |
| `GROQ_API_KEY`      | Yes*     | Yes                  | Groq API key                                                |
| `OPENAI_API_KEY`    | No       | -                    | OpenAI API key                                              |
| `ANTHROPIC_API_KEY` | No       | -                    | Anthropic API key                                           |
| `GOOGLE_API_KEY`    | No       | -                    | Google AI API key                                           |
| `LLM_PROVIDER`      | No       | `groq`             | LLM provider (`groq`,`openai`,`anthropic`,`google`) |
| `LLM_MODEL`         | No       | Auto                 | Specific model name                                         |
| `QDRANT_IN_MEMORY`  | No       | `false`            | Use in-memory vector store                                  |
| `QDRANT_URL`        | No       | `true`             | Qdrant Cloud URL                                            |
| `QDRANT_API_KEY`    | No       | `true`             | Qdrant Cloud API key                                        |
| `API_KEY`           | No       | -                    | Protect API with key                                        |
| `DEBUG`             | No       | `false`            | Enable debug mode                                           |
| `EMBEDDING_MODEL`   | No       | `all-MiniLM-L6-v2` | Sentence Transformers model                                 |
| `CHUNK_SIZE`        | No       | `512`              | Text chunk size                                             |
| `CHUNK_OVERLAP`     | No       | `50`               | Chunk overlap                                               |

*At least one LLM provider API key is required.

### Example .env File

```env
# LLM Provider (choose one)
GROQ_API_KEY=gsk_xxxxxxxxxxxx
# OPENAI_API_KEY=sk-xxxxxxxxxxxx
# ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxx

LLM_PROVIDER=groq

# Vector Store (optional - for persistence)
# QDRANT_URL=https://xxx.cloud.qdrant.io:6333
# QDRANT_API_KEY=xxxxxxxxxxxx
# QDRANT_IN_MEMORY=false

# Security (optional)
# API_KEY=your-secret-api-key

# Debug
DEBUG=false
```

## 📖 Usage

### Web Interface

1. **Upload Documents**
   * Click "Add source" in the left sidebar
   * Select PDF files (up to 50MB each)
   * Wait for processing to complete
2. **Select Sources**
   * Click document cards to select/deselect
   * Use "Select All" to query entire knowledge base
   * Selected documents shown in top bar
3. **Ask Questions**
   * Type your question in the chat input
   * Press Enter or click Send
   * View AI response with source citations
4. **View Sources**
   * Click source chips (e.g., "Page 7") to see context
   * Full page text with highlighted chunk
   * Relevance score and position info

### API Usage

```python
import requests

BASE_URL = "http://localhost:8000"

# Upload a document
with open("document.pdf", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/ingest",
        files={"file": f}
    )
    doc_id = response.json()["document_id"]

# Ask a question
response = requests.post(
    f"{BASE_URL}/query/ask",
    json={
        "question": "What is the main topic of this document?",
        "document_ids": [doc_id],
        "top_k": 5
    }
)
print(response.json()["answer"])

# Semantic search
response = requests.post(
    f"{BASE_URL}/query/search",
    json={
        "query": "machine learning",
        "top_k": 10
    }
)
for result in response.json()["results"]:
    print(f"Page {result['page_number']}: {result['text'][:100]}...")
```

## 📚 API Reference

### Ingestion Endpoints

#### `POST /ingest`

Upload and process a PDF document.

**Request:**

```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@document.pdf"
```

**Response:**

```json
{
  "document_id": "a4aee82f18829911",
  "filename": "document.pdf",
  "chunks": 42,
  "pages": 10,
  "message": "Document ingested: 42 chunks embedded and stored"
}
```

#### `GET /ingest/documents`

List all ingested documents.

**Response:**

```json
{
  "documents": [
    {"document_id": "a4aee82f18829911", "filename": "document.pdf"}
  ],
  "total": 1
}
```

#### `DELETE /ingest/{document_id}`

Delete a document and all its chunks.

#### `POST /ingest/clear`

Clear entire knowledge base.

### Query Endpoints

#### `POST /query/ask`

RAG question answering with source citations.

**Request:**

```json
{
  "question": "What are the key findings?",
  "document_ids": ["a4aee82f18829911"],
  "top_k": 5,
  "temperature": 0.3
}
```

**Response:**

```json
{
  "question": "What are the key findings?",
  "answer": "The key findings include...",
  "sources": [
    {
      "source_id": 1,
      "document_id": "a4aee82f18829911",
      "page_number": 3,
      "text_preview": "Our analysis reveals...",
      "relevance_score": 0.89,
      "char_start": 1250,
      "char_end": 1890
    }
  ],
  "model": "llama-3.3-70b-versatile"
}
```

#### `POST /query/search`

Semantic search across documents.

**Request:**

```json
{
  "query": "neural networks",
  "top_k": 10,
  "score_threshold": 0.5,
  "document_ids": null
}
```

#### `GET /ingest/documents/{document_id}/page/{page_number}/text`

Get full page text for highlighting.

### Health Endpoints

#### `GET /health`

Basic health check.

#### `GET /ready`

Readiness check with dependency status.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (index.html)                     │
│                    Modern UI with Chat Interface                 │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Backend                          │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   /ingest       │   /query        │   /health                   │
│   - Upload PDF  │   - /ask (RAG)  │   - Health check            │
│   - List docs   │   - /search     │   - Readiness               │
│   - Delete      │                 │                             │
└────────┬────────┴────────┬────────┴─────────────────────────────┘
         │                 │
         ▼                 ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐
│ Document        │ │ Retriever       │ │ Generator               │
│ Processor       │ │                 │ │                         │
│ - PDF Extract   │ │ - Embed query   │ │ - Build prompt          │
│ - Chunking      │ │ - Vector search │ │ - LLM completion        │
│ - Position track│ │ - Filter docs   │ │ - Citation extraction   │
└────────┬────────┘ └────────┬────────┘ └────────┬────────────────┘
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐
│ Embedding       │ │ Vector Store    │ │ LLM Providers           │
│ Service         │ │ (Qdrant)        │ │                         │
│                 │ │                 │ │ - Groq (default)        │
│ Sentence        │ │ - In-memory     │ │ - OpenAI                │
│ Transformers    │ │ - Qdrant Cloud  │ │ - Anthropic             │
│                 │ │ - Self-hosted   │ │ - Google AI             │
└─────────────────┘ └─────────────────┘ └─────────────────────────┘
```

### Data Flow

1. **Ingestion Pipeline**
   ```
   PDF → Extract Text → Chunk with Overlap → Embed → Store in Qdrant
   ```
2. **Query Pipeline**
   ```
   Question → Embed → Vector Search → Retrieve Context → LLM Generate → Response
   ```

### Project Structure

```
enterprise-rag/
├── app/
│   ├── api/
│   │   └── routes/
│   │       ├── health.py      # Health endpoints
│   │       ├── ingest.py      # Document ingestion
│   │       └── query.py       # Search and RAG
│   ├── core/
│   │   ├── document_processor.py  # PDF processing
│   │   ├── embeddings.py          # Embedding service
│   │   ├── generator.py           # LLM generation
│   │   ├── retriever.py           # Semantic search
│   │   └── vector_store.py        # Qdrant operations
│   ├── models/
│   │   └── schemas.py         # Pydantic models
│   ├── static/
│   │   └── index.html         # Frontend UI
│   ├── config.py              # Settings
│   ├── exceptions.py          # Error handlers
│   ├── middleware.py          # Auth, logging
│   └── utils.py               # Utilities
├── main.py                    # Application entry
├── Dockerfile
├── requirements.txt
└── README.md
```

## 🚀 Deployment

### Hugging Face Spaces (Free)

1. Create account at https://huggingface.co
2. Get Groq API key at https://console.groq.com
3. Create new Space with Docker SDK
4. Upload project files
5. Add `GROQ_API_KEY` secret
6. Wait for build (~5 minutes)

See [Deployment Guide](https://claude.ai/chat/docs/DEPLOYMENT.md) for detailed instructions.

### Docker (Self-hosted)

```bash
docker build -t enterprise-rag .
docker run -d \
  -p 7860:7860 \
  -e GROQ_API_KEY=your_key \
  -e QDRANT_URL=https://your-qdrant.cloud.qdrant.io:6333 \
  -e QDRANT_API_KEY=your_qdrant_key \
  enterprise-rag
```

## *Kubernetes : Comming Soon*

## 🔧 Development

### Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# With coverage
pytest --cov=app --cov-report=html
```

### Code Quality

```bash
# Format code
ruff format .

# Lint
ruff check .

# Type checking
mypy app/
```

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://claude.ai/chat/LICENSE) file for details.

## 🙏 Acknowledgments

* [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
* [Qdrant](https://qdrant.tech/) - Vector database
* [Sentence Transformers](https://www.sbert.net/) - Embeddings
* [LangChain](https://langchain.com/) - LLM framework
* [Groq](https://groq.com/) - Fast LLM inference

---

<p align="center">
  Made with ❤️ for the AI community
</p>
