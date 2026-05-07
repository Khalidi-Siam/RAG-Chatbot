# Cognivue — RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot backend built with FastAPI. Cognivue allows users to upload PDF documents and ask natural-language questions against them. It maintains per-session knowledge bases and conversational memory, delivering grounded, source-cited answers powered by Google Gemini.

---

## Table of Contents

- [Overview](#overview)
- [Architecture & Pipeline](#architecture--pipeline)
  - [Ingestion Pipeline](#ingestion-pipeline)
  - [Query Pipeline](#query-pipeline)
- [System Prompt & Reasoning Logic](#system-prompt--reasoning-logic)
  - [Case 1 — Unresolvable Reference, No History](#case-1--unresolvable-reference-no-history)
  - [Case 2 — Resolvable Reference via Chat History](#case-2--resolvable-reference-via-chat-history)
  - [Case 3 — Context Insufficient, History Fills the Gap](#case-3--context-insufficient-history-fills-the-gap)
  - [Case 4 — Neither Context Nor History Sufficient](#case-4--neither-context-nor-history-sufficient)
  - [Case 5 — Normal Answer from Context](#case-5--normal-answer-from-context)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Database Models](#database-models)
- [Session & Memory Management](#session--memory-management)
- [Configuration](#configuration)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
  - [Session Management](#session-management)
  - [Knowledge Base](#knowledge-base)
  - [Chat](#chat)
  - [Error Schema](#error-schema)
- [Frontend Integration Notes](#frontend-integration-notes)

---

## Overview

Cognivue is a **stateless RAG backend**. Each HTTP request is self-contained — session state and conversational memory are managed externally via:

- **PostgreSQL** — persists session metadata, knowledge base paths, and document records.
- **Upstash Redis** — stores in-memory chat history with automatic TTL expiry.
- **FAISS (on-disk)** — stores and retrieves per-session vector embeddings.

This separation of concerns allows the core RAG pipeline to remain purely functional (no internal state), making it easily scalable and testable.

---

## Architecture & Pipeline

### Ingestion Pipeline

When a PDF is uploaded, the following sequence runs synchronously:

```
PDF File
  │
  ▼
PDFIngestor (PyMuPDF)
  │  - Extracts text page-by-page
  │  - Cleans whitespace and formatting artifacts
  │  - Splits pages into chunks via LangChain's RecursiveCharacterTextSplitter
  │    (chunk_size=1000, chunk_overlap=150)
  │  - Each chunk is tagged with: chunk_id (UUID), page number, source filename
  ▼
GeminiEmbedder
  │  - Embeds all chunk texts using Google's gemini-embedding-001 model
  ▼
FAISSVectorStore (session-specific)
  │  - Normalizes embeddings (L2) for cosine similarity via IndexFlatIP
  │  - Persists the FAISS index (.index) and aligned metadata (.json) to disk
  │  - Path: faiss_db/session_{session_id}/pdf_knowledge.index
  ▼
PostgreSQL (Document row updated)
  │  - status: "processing" → "indexed"
  │  - total_pages, total_chunks recorded
```

A **threading lock** (via `LockManager`) is acquired per session before any FAISS write, preventing race conditions when multiple files are uploaded concurrently to the same session.

---

### Query Pipeline

When a question is submitted to the chat endpoint:

```
User Question
  │
  ▼
chat_routes.py
  │  1. Validates session (active in PostgreSQL)
  │  2. Checks if any documents are indexed (no-doc guard)
  │  3. Fetches last 16 messages from Redis (session_manager.get_history)
  ▼
RAGPipeline.ask()
  │
  ├── GeminiEmbedder.embed_query(question)
  │     - Embeds the raw user question
  │
  ├── Retriever.retrieve(query_embedding, top_k=5)
  │     - Loads session FAISS index from disk
  │     - Runs similarity search (cosine), returns top-k chunks
  │     - Filters by similarity_threshold=0.6
  │
  ├── Build context string
  │     - Each chunk formatted as: "(Page N) <chunk text>"
  │
  ├── Format chat history string
  │     - Each message formatted as: "ROLE: <message>"
  │
  ├── build_rag_prompt(context, question, chat_history)
  │     - Constructs the full prompt (see System Prompt section)
  │
  └── GeminiLLM.generate(prompt)
        - Calls gemini-2.5-flash with the assembled prompt
        - Returns the model's text answer
  │
  ▼
chat_routes.py
  │  - Saves user message + assistant answer to Redis
  │  - Returns: { session_id, answer, sources[] }
```

---

## System Prompt & Reasoning Logic

The system prompt is assembled dynamically by `build_rag_prompt()` in `src/components/prompt_template.py`. It is structured in three sections:

```
[CHAT HISTORY]   ← only included if prior messages exist
[CONTEXT]        ← retrieved chunks from FAISS (always present)
[INSTRUCTIONS]   ← ordered rules for the LLM to follow
[QUESTION]
ANSWER:
```

The instructions encode a **priority-ordered decision tree** that the LLM must follow. Here is the full logic and the real-world cases it handles:

---

### Case 1 — Unresolvable Reference, No History

**Trigger:** The question contains a pronoun or vague reference (e.g., *"that"*, *"it"*, *"the incident"*, *"he"*, *"those"*) **and** there is no chat history to resolve what it refers to.

**Example:**
```
User: "What was the year of that incident?"
(First message in session — no history)
```

**Behavior:** The model does **not** attempt to guess. It responds with a clarification request:
> *"Your question refers to something (e.g. 'that incident') that hasn't been identified. Could you clarify what you are referring to?"*

**Why:** Without history, the pronoun is ambiguous. Answering would require hallucination. The prompt explicitly instructs the model to catch this before looking at context or generating a response.

---

### Case 2 — Resolvable Reference via Chat History

**Trigger:** The question contains a vague reference **but** the chat history clearly resolves what it refers to.

**Example:**
```
User: "What is the Permanent Settlement Act?"
Assistant: "Under Lord Cornwallis, the British introduced the Permanent Settlement Act of 1793..."

User: "What was the year of that incident?"
```

**Behavior:** The model identifies *"that incident"* → *Permanent Settlement Act* from the history, substitutes the resolved entity, and answers:
> *"The Permanent Settlement Act was introduced in 1793."*

**Why:** Chat history is included in the prompt at the top. The instruction tells the model to use it to resolve references before proceeding to answer. No secondary API call for query reformulation is needed — the LLM handles this natively within a single prompt.

---

### Case 3 — Context Insufficient, History Fills the Gap

**Trigger:** The retrieved FAISS chunks do not contain enough information to answer the question fully, but the conversation history contains relevant content.

**Example:**
```
User: "Summarize the whole chat."
Assistant: "The chat is about permanent settlement act...."

```
The FAISS retriever may return chunks that will not relevant to chat summary. The model uses the chat history and give summary based on the history.

**Behavior:** The model **combines** the retrieved context and the chat history to form a complete, coherent answer.

**Why:** Rule 5 in the prompt explicitly instructs the model to fall back to chat history as a supplementary source when context alone is insufficient. This prevents unnecessarily vague or incomplete answers.

---

### Case 4 — Neither Context Nor History Sufficient

**Trigger:** The question asks about something that is simply not present in the uploaded documents, and the conversation history offers no additional insight.

**Example:**
```
User: "What is the capital of France?"
(Document is about the Permanent Settlement Act — no relevant context retrieved)
```

**Behavior:** The model responds with a grounded refusal:
> *"Not found in the knowledge base."*

**Why:** Rules 6–8 in the prompt collectively prevent hallucination. The model is explicitly told not to use external knowledge and not to guess. This keeps the chatbot grounded strictly to the uploaded documents.

---

### Case 5 — Normal Answer from Context

**Trigger:** The retrieved chunks contain sufficient information to answer the question directly.

**Behavior:** The model answers using the provided context, citing the information from the retrieved page chunks. This is the standard, happy-path flow.

---

### Prompt Structure (Annotated)

```
CHAT HISTORY (prior conversation turns for reference):
USER: <message>
ASSISTANT: <message>
...

CONTEXT (retrieved from the knowledge base):
(Page 1) <chunk text>
(Page 2) <chunk text>
...

INSTRUCTIONS:
You are a knowledgeable assistant. Answer the user's question by following these rules in order:
1. Check for unresolved references (that, it, this, those, the incident, the event, he, she, they, etc.)
2. If references exist and NO chat history can resolve them → ask for clarification.
3. If references exist and chat history RESOLVES them → substitute and proceed.
4. Use the CONTEXT as primary source of information.
5. If context is insufficient but CHAT HISTORY has relevant info → combine both.
6. If neither context nor history is sufficient → respond "Not found in the knowledge base."
7. Do not use external knowledge beyond what is provided.
8. Do not guess or hallucinate facts.
9. Keep the answer clear and concise.

QUESTION:
<user question>

ANSWER:
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API Framework** | FastAPI |
| **LLM** | Google Gemini (`gemini-2.5-flash`) |
| **Embeddings** | Google Gemini (`gemini-embedding-001`) |
| **Vector Store** | FAISS (`faiss-cpu`) — per-session, on-disk |
| **Chat Memory** | Upstash Redis (HTTP REST client) |
| **Relational DB** | PostgreSQL via SQLAlchemy |
| **PDF Processing** | PyMuPDF (`fitz`) |
| **Text Chunking** | LangChain `RecursiveCharacterTextSplitter` |
| **Settings** | Pydantic `BaseSettings` |

---

## Project Structure

```
RAG-Chatbot/
├── requirements.txt
├── setup.py
├── .env                      # Environment variables (not committed)
└── src/
    ├── api/
    │   ├── main.py           # FastAPI app, CORS, router registration
    │   └── routes/
    │       ├── session_routes.py   # /session/start, /end, /messages, /documents
    │       ├── upload_routes.py    # /session/{id}/upload
    │       └── chat_routes.py      # /session/{id}/chat
    ├── components/
    │   ├── pdf_ingestor.py         # PDF extraction & chunking (PyMuPDF + LangChain)
    │   ├── embedder.py             # GeminiEmbedder (documents & queries)
    │   ├── faiss_store.py          # FAISSVectorStore (add, query, persist, reset)
    │   ├── retriever.py            # Retriever (similarity threshold filtering)
    │   ├── gemini_llm.py           # GeminiLLM (text generation)
    │   ├── prompt_template.py      # build_rag_prompt() — system prompt assembly
    │   └── chroma_store.py         # (ChromaDB alternative, currently unused)
    ├── config/
    │   └── settings.py             # Pydantic Settings (env vars, defaults)
    ├── db/
    │   ├── database.py             # SQLAlchemy engine & session factory
    │   ├── init_db.py              # Table creation helper
    │   └── models.py               # ORM models: Session, KnowledgeBase, Document
    ├── pipeline/
    │   └── rag_pipeline.py         # RAGPipeline (ingest_pdf, ask)
    ├── services/
    │   ├── session_manager.py      # RedisSessionManager (add, get, delete messages)
    │   └── lock_manager.py         # Per-session threading locks for FAISS writes
    ├── data/                       # (reserved for local data)
    ├── tests/                      # Test suite
    ├── exception.py                # Custom exception classes
    └── logger.py                   # Centralized logging setup
```

---

## Database Models

Three PostgreSQL tables manage session state:

**`sessions`**
| Column | Type | Description |
|---|---|---|
| `id` | UUID (PK) | Unique session identifier |
| `is_active` | Boolean | Whether the session is currently active |
| `created_at` | DateTime | Session creation timestamp |
| `last_activity` | DateTime | Updated on every chat interaction |
| `ended_at` | DateTime (nullable) | Set when the session is ended |

**`knowledge_bases`** *(1-to-1 with sessions)*
| Column | Type | Description |
|---|---|---|
| `id` | UUID (PK) | Knowledge base identifier |
| `session_id` | UUID (FK) | Foreign key to `sessions` |
| `upload_path` | Text | Filesystem path for raw PDF uploads |
| `faiss_path` | Text | Filesystem path for FAISS index files |

**`documents`** *(many-to-1 with knowledge_bases)*
| Column | Type | Description |
|---|---|---|
| `id` | UUID (PK) | Document identifier |
| `knowledge_base_id` | UUID (FK) | Foreign key to `knowledge_bases` |
| `original_filename` | Text | Original uploaded filename |
| `stored_filename` | Text | Saved filename on disk |
| `file_path` | Text | Full path to the saved file |
| `file_size` | Integer | File size in bytes |
| `total_pages` | Integer | Pages extracted |
| `total_chunks` | Integer | Chunks generated and indexed |
| `status` | String | `"processing"` → `"indexed"` (or `"failed"`) |
| `uploaded_at` | DateTime | Upload timestamp |

> Cascading deletes are configured: ending a session deletes its knowledge base and all associated document records.

---

## Session & Memory Management

Chat history is stored in **Upstash Redis** using the key pattern `session:{session_id}:messages`, which holds a JSON array of message objects.

**Key behaviors:**
- **TTL Refresh:** The TTL (default: 1 hour) is reset on every `add_message` and `get_history` call. As long as the user remains active, the session stays alive.
- **Window Limit for RAG:** The chat endpoint fetches the **last 16 messages** (8 turns) to inject into the RAG prompt, keeping the prompt size manageable.
- **Full History Retrieval:** The `/messages` endpoint fetches the last 50 messages for frontend display.
- **Cleanup:** Calling `/session/{session_id}/end` deletes the Redis key, FAISS index, uploaded files, and all database rows for the session.

---

## Configuration

All settings are managed via Pydantic `BaseSettings` and loaded from the `.env` file:

```env
# Required
GOOGLE_API_KEY=your_gemini_api_key
DATABASE_URL=postgresql://user:password@host:port/dbname

# Upstash Redis
UPSTASH_REDIS_REST_URL=https://your-redis-url.upstash.io
UPSTASH_REDIS_REST_TOKEN=your_redis_token

# CORS (comma-separated)
CORS_ORIGINS=http://localhost:5173

# Optional overrides (defaults shown)
TTL_SECONDS=3600
EMBEDDING_MODEL=gemini-embedding-001
LLM_MODEL=gemini-2.5-flash
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
SIMILARITY_THRESHOLD=0.6
TOP_K=5
FAISS_BASE_DIR=faiss_db
UPLOAD_DIR=uploads
```

---

## Getting Started

### Prerequisites
- Python 3.9+
- PostgreSQL database
- Upstash Redis account (free tier available at [upstash.com](https://upstash.com))
- Google Gemini API Key

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Khalidi-Siam/RAG-Chatbot.git
   cd RAG-Chatbot
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate       # Linux / macOS
   venv\Scripts\activate          # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # or, for editable install:
   pip install -e .
   ```

4. **Create your `.env` file** in the project root and fill in all required variables (see [Configuration](#configuration)).

5. **Initialize the database:**
   ```bash
   python -c "from src.db.init_db import init; init()"
   ```

### Running the Server

```bash
uvicorn src.api.main:app --reload
```

The API will be available at `http://localhost:8000`.  
Interactive Swagger UI: `http://localhost:8000/docs`  
ReDoc: `http://localhost:8000/redoc`

---

## API Reference

**Base URL:** `http://localhost:8000`

All session-scoped endpoints use the path prefix `/session`.

---

### Session Management

#### `POST /session/start`

Creates a new anonymous session. Sets up the FAISS and upload folder paths, creates the `sessions` and `knowledge_bases` database rows.

**Response `200`:**
```json
{
  "message": "Session created successfully",
  "session_id": "eae7647c-ba53-4e34-a669-8872598d8ebe",
  "upload_path": "uploads/session_eae7647c-...",
  "faiss_path": "faiss_db/session_eae7647c-..."
}
```

---

#### `POST /session/{session_id}/end`

Ends the session. Deletes:
- All uploaded PDF files from disk
- The FAISS index directory
- All PostgreSQL rows (session, knowledge base, documents) via cascade
- The Redis chat history key

**Path Parameters:**
| Parameter | Type | Description |
|---|---|---|
| `session_id` | string (UUID) | The session to end |

**Response `200`:**
```json
{
  "message": "Session ended and cleaned up successfully."
}
```

**Error `404`:** Session not found.

---

#### `GET /session/{session_id}/messages`

Retrieves the full chat history for a session (up to the last 50 messages) from Redis.

**Path Parameters:**
| Parameter | Type | Description |
|---|---|---|
| `session_id` | string (UUID) | The target session |

**Response `200`:**
```json
{
  "session_id": "eae7647c-ba53-4e34-a669-8872598d8ebe",
  "messages": [
    { "role": "user", "message": "What is permanent settlement act?" },
    { "role": "assistant", "message": "Under Lord Cornwallis, the British introduced the Permanent Settlement Act of 1793..." },
    { "role": "user", "message": "What was the year of that incident?" },
    { "role": "assistant", "message": "The Permanent Settlement Act was introduced in 1793." }
  ]
}
```

**Empty session response:**
```json
{
  "session_id": "...",
  "messages": [],
  "message": "No messages found for this session."
}
```

---

#### `GET /session/{session_id}/documents`

Returns the list of all documents uploaded to the session's knowledge base, including their metadata from PostgreSQL.

**Path Parameters:**
| Parameter | Type | Description |
|---|---|---|
| `session_id` | string (UUID) | The target session |

**Response `200`:**
```json
{
  "session_id": "eae7647c-ba53-4e34-a669-8872598d8ebe",
  "documents": [
    {
      "id": "31945e69-0a da-4f0b-8c25-fedb2d608c8c",
      "filename": "sample1.pdf",
      "file_size": 204800,
      "total_pages": 12,
      "status": "indexed",
      "uploaded_at": "2026-05-07T15:33:00Z"
    }
  ]
}
```

**Error `404`:** Session not found.

---

### Knowledge Base

#### `POST /session/{session_id}/upload`

Uploads a PDF document and ingests it into the session's FAISS knowledge base. The pipeline runs synchronously — the response is returned only after indexing is complete.

**Path Parameters:**
| Parameter | Type | Description |
|---|---|---|
| `session_id` | string (UUID) | Target session |

**Request:** `multipart/form-data`
| Field | Type | Description |
|---|---|---|
| `file` | binary | The PDF file to upload |

> ⚠️ **Do not manually set `Content-Type`** when using `fetch`/`axios`. Let the browser set the multipart boundary automatically.

**Response `200`:**
```json
{
  "message": "PDF uploaded and indexed successfully.",
  "document_id": "c74daa6b-f33e-4ea3-b24e-0260721b370a",
  "filename": "sample1.pdf",
  "file_size": 204800,
  "total_pages": 12,
  "total_chunks": 47,
  "status": "indexed"
}
```

**Errors:**
| Code | Reason |
|---|---|
| `400` | File is not a PDF |
| `404` | Session not found or inactive |
| `500` | FAISS ingestion failure |

---

### Chat

#### `POST /session/{session_id}/chat`

Asks a question against the session's knowledge base. The pipeline retrieves relevant chunks, injects the last 16 messages of chat history, builds the system prompt, and queries Gemini.

**Path Parameters:**
| Parameter | Type | Description |
|---|---|---|
| `session_id` | string (UUID) | Target session |

**Request Body:** `application/json`
```json
{
  "question": "What was the social impact of the Permanent Settlement Act?"
}
```

**Response `200`:**
```json
{
  "session_id": "eae7647c-ba53-4e34-a669-8872598d8ebe",
  "answer": "Under Lord Cornwallis, the British introduced the Permanent Settlement Act of 1793. This act created a new class of hereditary landlords (Zamindars) who owned the land as long as they paid a fixed revenue to the British. Its social impact was that the traditional peasantry (ryots) lost their rights to the land and were often exploited, leading to a persistent socioeconomic divide.",
  "sources": [
    {
      "page": 1,
      "source_file": "sample1.pdf",
      "chunk_id": "d2f36216-99ce-4e05-870c-9ea369faddc3",
      "cosine_similarity": 0.7267
    },
    {
      "page": 1,
      "source_file": "sample1.pdf",
      "chunk_id": "e0deb0d3-7098-481f-b41e-5e6810a35c5b",
      "cosine_similarity": 0.6697
    },
    {
      "page": 2,
      "source_file": "sample1.pdf",
      "chunk_id": "81980e2e-a374-4315-8df7-f5aaa37adde8",
      "cosine_similarity": 0.6246
    }
  ]
}
```

**Special case — no documents uploaded:**
```json
{
  "session_id": "...",
  "answer": "Please upload a document first to start chatting.",
  "sources": []
}
```

**Errors:**
| Code | Reason |
|---|---|
| `404` | Session not found or inactive |
| `500` | RAG pipeline or LLM failure |

---

### Error Schema

Validation errors (`422 Unprocessable Entity`) follow the standard FastAPI format:

```json
{
  "detail": [
    {
      "loc": ["body", "question"],
      "msg": "field required",
      "type": "value_error.missing",
      "input": {},
      "ctx": {}
    }
  ]
}
```

---

## Frontend Integration Notes

- **Store the `session_id`** returned from `POST /session/start` in `sessionStorage` (clears on browser close) or `localStorage` (persists). Use it as a path parameter in every subsequent call.
- **Do not set `Content-Type` manually** for the upload endpoint. When using `FormData` with `fetch`, the browser automatically adds the correct `multipart/form-data; boundary=...` header.
- **Resync on page refresh** by calling `GET /session/{session_id}/messages` to repopulate the chat UI and `GET /session/{session_id}/documents` to repopulate the knowledge base sidebar.
- **Session lifetime** is governed by Redis TTL (default: 1 hour of inactivity). If the Redis key expires, the session's chat memory is lost even if the database rows remain. Always handle 404s gracefully and prompt the user to start a new session.
- **Source citations** in the `/chat` response include `page`, `source_file`, and `cosine_similarity`, which can be used to render source attribution in the UI.
