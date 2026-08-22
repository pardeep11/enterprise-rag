# Enterprise RAG

An end-to-end Enterprise Retrieval-Augmented Generation (RAG) system built with **Python, LlamaIndex, FAISS CPU, Ollama, and FastAPI**.

The system processes enterprise PDF documents, creates semantic chunks, generates embeddings using Ollama, stores vectors in FAISS, performs semantic retrieval, applies metadata filtering, and uses a local Ollama LLM to generate grounded answers.

The project is designed as a modular foundation for building production-oriented Enterprise RAG systems.

---

# 🚀 Project Status

| Milestone | Description            | Status     |
| --------- | ---------------------- | ---------- |
| 1         | Project Setup          | ✅ Complete |
| 2         | PDF Document Loading   | ✅ Complete |
| 3         | Document Chunking      | ✅ Complete |
| 4         | Embedding Generation   | ✅ Complete |
| 5         | FAISS Vector Indexing  | ✅ Complete |
| 6         | Semantic Search        | ✅ Complete |
| 7         | Ollama LLM Integration | ✅ Complete |
| 8         | Complete RAG Pipeline  | ✅ Complete |
| 9         | FastAPI APIs           | ✅ Complete |
| 10        | Metadata Filtering     | ✅ Complete |

### Current Progress

**10/10 milestones completed 🎉**

---

# 🏗️ Architecture

```text
                                        ┌─────────────────────────┐
                    │   Enterprise Documents  │
                    │       PDF Files         │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   LlamaIndex Loader     │
                    │   PDF + Metadata        │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │       Chunking           │
                    │   SentenceSplitter       │
                    │  chunk=256, overlap=30   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Embedding Generation   │
                    │      Ollama              │
                    │   nomic-embed-text       │
                    │      768 dimensions      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     FAISS Vector Store   │
                    │       IndexFlatL2        │
                    │                          │
                    │     index.faiss          │
                    │     metadata.pkl         │
                    └────────────┬────────────┘
                                 │
                                 │
                    ┌────────────▼────────────┐
                    │       User Question      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Question Embedding    │
                    │        Ollama            │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    FAISS Semantic Search │
                    │                          │
                    │   Candidate Retrieval    │
                    │       Top-K × 4          │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Metadata Filtering     │
                    │                          │
                    │   file_name              │
                    │   department             │
                    │   page_label             │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Top-K Relevant Chunks  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Context Construction   │
                    │           +             │
                    │    Prompt Construction  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │       Ollama LLM         │
                    │      Qwen / Llama        │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    Answer + Sources      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │        FastAPI           │
                    │                          │
                    │  GET  /health            │
                    │  POST /ingest            │
                    │  POST /search            │
                    │  POST /ask               │
                    └─────────────────────────┘

                    
```

---

# 🔄 RAG Pipeline

The complete retrieval and generation workflow:

```text
User Question
      │
      ▼
Question Embedding
      │
      ▼
FAISS Semantic Search
      │
      ▼
Candidate Retrieval
      │
      ▼
Metadata Filtering
      │
      ▼
Top-K Relevant Chunks
      │
      ▼
Context Construction
      │
      ▼
Prompt Construction
      │
      ▼
Ollama LLM
      │
      ▼
Generated Answer
      │
      ▼
Answer + Sources
```

---

# 🧰 Tech Stack

## Backend

* Python
* FastAPI
* Uvicorn
* Pydantic

## RAG / AI

* LlamaIndex
* FAISS CPU
* Ollama
* Nomic Embed Text
* Qwen / Llama

## Data Processing

* LlamaIndex PDF Loader
* LlamaIndex `SentenceSplitter`
* NumPy
* Pickle

## Infrastructure

* Docker
* Docker Compose

> **Note:** PyMuPDF is **not used** in this project.

---

# 📂 Project Structure

```text
enterprise-rag/
│
├── Dockerfile
├── README.md
├── requirements.txt
├── docker-compose.yml
│
├── app/
│   ├── __init__.py
│   │
│   ├── api/
│   │
│   ├── chunking/
│   │   ├── __init__.py
│   │   └── chunker.py
│   │
│   ├── config/
│   │
│   ├── core/
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedder.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   └── ollama_llm.py
│   │
│   ├── loaders/
│   │   └── pdf_loader.py
│   │
│   ├── main.py
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── rag_pipeline.py
│   │   └── test_rag.py
│   │
│   ├── retrieval/
│   │   ├── search.py
│   │   └── test_search.py
│   │
│   ├── schemas/
│   │
│   ├── services/
│   │
│   ├── utils/
│   │
│   └── vectorstore/
│       ├── __init__.py
│       └── index.py
│
├── data/
│   ├── faiss/
│   │
│   ├── pdf/
│   │   ├── cloud/
│   │   │   ├── DigitalOcean_policies.pdf
│   │   │   └── aws_policy.pdf
│   │   │
│   │   ├── hr/
│   │   │   └── Leave_Policy.pdf
│   │   │
│   │   └── travel/
│   │       └── travel_policy.pdf
│   │
│   └── processed/
│
├── docs/
│
├── logs/
│
├── tests/
│
├── vectorstore/
│   ├── index.faiss
│   └── metadata.pkl
│
├── pass
├── pass.pub
└── project_backup.zip
```

---

# 📌 Milestone 1 — Project Setup

Created the initial modular Enterprise RAG project structure.

The application is separated into dedicated modules for:

* API layer
* Configuration
* Core functionality
* PDF loading
* Document chunking
* Embedding generation
* LLM integration
* Retrieval
* Vector storage
* RAG pipeline
* Schemas
* Services
* Utilities
* Testing

This modular structure makes the project easier to maintain and extend.

---

# 📄 Milestone 2 — PDF Document Loading

PDF documents are loaded using **LlamaIndex**.

Current enterprise documents include:

```text
data/pdf/

├── cloud/
│   ├── DigitalOcean_policies.pdf
│   └── aws_policy.pdf
│
├── hr/
│   └── Leave_Policy.pdf
│
└── travel/
    └── travel_policy.pdf
```

The loader extracts document content along with document metadata.

Example metadata:

```text
file_name
file_path
file_type
page_label
file_size
creation_date
last_modified_date
```

The metadata is preserved and later used for filtering and source attribution.

---

# ✂️ Milestone 3 — Document Chunking

Documents are divided into smaller chunks using LlamaIndex `SentenceSplitter`.

Current configuration:

```python
SentenceSplitter(
    chunk_size=256,
    chunk_overlap=30,
)
```

The purpose of chunking is to:

* Reduce the size of retrieved context
* Improve semantic retrieval
* Create meaningful embedding units
* Prevent unnecessarily large prompts

Pipeline:

```text
PDF
 ↓
Documents
 ↓
Sentences / Text
 ↓
Chunks
```

---

# 🧠 Milestone 4 — Embedding Generation

The project uses Ollama for local embedding generation.

Embedding model:

```text
nomic-embed-text
```

Current embedding dimension:

```text
768
```

Embedding pipeline:

```text
Text Chunk
    ↓
Ollama
    ↓
nomic-embed-text
    ↓
768-dimensional vector
```

The generated vectors are used for semantic retrieval.

---

# 🗄️ Milestone 5 — FAISS Vector Indexing

FAISS CPU is used as the vector search engine.

The generated embeddings are stored in a FAISS index.

Current vector store:

```text
vectorstore/

├── index.faiss
└── metadata.pkl
```

Current index information:

```text
FAISS vectors : 91
Dimensions    : 768
Index type    : IndexFlatL2
```

The architecture separates vector storage from document metadata.

```text
index.faiss
     ↓
Vector representation

metadata.pkl
     ↓
Chunk text + metadata
```

FAISS vector IDs are used to map retrieved vectors back to the original document chunks.

---

# 🔎 Milestone 6 — Semantic Search

Semantic search retrieves document chunks based on vector similarity rather than exact keyword matching.

Search flow:

```text
User Question
      ↓
Question Embedding
      ↓
FAISS Search
      ↓
Relevant Candidate Chunks
```

Example:

```text
Question:
What is the maximum casual leave allowed per month?
```

The question is converted into an embedding and compared against document embeddings stored in FAISS.

The search results contain:

* Rank
* Distance score
* Text
* Metadata

### FAISS Distance

The current index uses:

```text
IndexFlatL2
```

Therefore the returned score represents **L2 distance**.

Generally:

```text
Lower distance  → More similar
Higher distance → Less similar
```

The distance should not be interpreted as a percentage similarity score.

---

# 🤖 Milestone 7 — Ollama LLM Integration

Ollama is used to run the LLM locally.

LLM integration is implemented in:

```text
app/llm/ollama_llm.py
```

Architecture:

```text
RAG Application
      ↓
Ollama
      ↓
Local LLM
      ↓
Generated Response
```

The project can use local models such as:

```text
Qwen
Llama
```

This allows the RAG system to generate answers without requiring a hosted LLM API for the generation step.

---

# 🧩 Milestone 8 — Complete RAG Pipeline

The complete RAG pipeline combines retrieval and generation.

Implementation:

```text
app/rag/rag_pipeline.py
```

Pipeline:

```text
User Question
      ↓
Question Embedding
      ↓
FAISS Semantic Search
      ↓
Retrieve Relevant Chunks
      ↓
Build Context
      ↓
Construct Prompt
      ↓
Ollama LLM
      ↓
Generate Answer
      ↓
Return Answer + Sources
```

Example:

```json
{
    "question": "What is the maximum casual leave allowed per month?",
    "answer": "The maximum casual leave allowed per month is a total of 3 days.",
    "sources": [
        {
            "rank": 1,
            "score": 0.46,
            "text": "...",
            "metadata": {
                "file_name": "Leave_Policy.pdf",
                "page_label": "2"
            }
        }
    ]
}
```

The retrieved sources provide transparency and allow users to identify where the answer originated.

---

# 🌐 Milestone 9 — FastAPI APIs

The RAG system is exposed through FastAPI REST APIs.

The API layer provides access to ingestion, retrieval, health checking, and question answering.

Current API operations include:

```text
GET  /health
POST /ingest
POST /search
POST /ask
```

---

## Health Check

```text
GET /health
```

Used to verify that the API service is running.

Example:

```json
{
    "status": "healthy"
}
```

---

## Document Ingestion

```text
POST /ingest
```

The ingestion endpoint processes the configured documents and creates/updates the vector index.

Pipeline:

```text
PDF Documents
      ↓
Load
      ↓
Chunk
      ↓
Embed
      ↓
FAISS
      ↓
Metadata
```

---

## Semantic Search

```text
POST /search
```

The search endpoint performs semantic retrieval and supports metadata filtering.

Example concept:

```text
Question
+
Metadata Filters
        ↓
Semantic Search
        ↓
Filtered Results
```

---

## RAG Question Answering

```text
POST /ask
```

The endpoint executes the complete RAG pipeline.

Example request:

```json
{
    "question": "What is the maximum casual leave allowed per month?"
}
```

Example response:

```json
{
    "question": "What is the maximum casual leave allowed per month?",
    "answer": "The maximum casual leave allowed per month is a total of 3 days.",
    "sources": [
        {
            "rank": 1,
            "score": 0.46,
            "text": "...",
            "metadata": {
                "file_name": "Leave_Policy.pdf",
                "page_label": "2"
            }
        }
    ]
}
```

---

# 🔐 Milestone 10 — Metadata Filtering

Metadata filtering has been implemented as part of the retrieval layer.

Implementation:

```text
app/retrieval/search.py
```

The retrieval system supports:

```text
file_name
department
page_label
```

This allows semantic search to be combined with structured document filtering.

---

## Metadata Filtering Flow

```text
User Question
      ↓
Question Embedding
      ↓
FAISS Search
      ↓
Candidate Retrieval
      ↓
File Name Filter
      ↓
Department Filter
      ↓
Page Filter
      ↓
Top-K Results
```

---

## Example

Question:

```text
What is earned leave?
```

Filters:

```text
file_name  = Leave_Policy.pdf
department = hr
page_label = 2
```

The system first retrieves semantically similar candidates and then checks whether each candidate satisfies the requested metadata filters.

---

## Candidate Retrieval

The system retrieves more candidates than the requested final `top_k`.

Example:

```python
candidate_k = top_k * 4
```

For:

```text
top_k = 5
```

the system retrieves:

```text
20 candidates
```

before applying metadata filters.

```text
FAISS Search
     ↓
20 Candidates
     ↓
Metadata Filtering
     ↓
Valid Candidates
     ↓
Top 5 Results
```

This improves retrieval robustness when some of the highest-ranked semantic results do not satisfy the requested filters.

---

## Metadata Filter Logic

### File Name

Example:

```text
Leave_Policy.pdf
```

Only chunks from the requested file are returned.

### Department

The department is derived from the document path.

Example:

```text
data/pdf/hr/Leave_Policy.pdf
```

The department is:

```text
hr
```

### Page

The search can be restricted to a specific page:

```text
page_label = 2
```

All supplied filters are combined using AND logic.

Example:

```text
file_name = Leave_Policy.pdf
AND
department = hr
AND
page_label = 2
```

A result must satisfy all supplied filters to be returned.

---

# 📡 API Request Example

## Basic Search

```json
{
    "question": "What is the maximum casual leave allowed per month?"
}
```

## Search With Metadata Filters

```json
{
    "question": "What is earned leave?",
    "filters": {
        "file_name": "Leave_Policy.pdf",
        "department": "hr",
        "page_label": "2"
    }
}
```

Example result structure:

```json
{
    "question": "What is earned leave?",
    "filters": {
        "file_name": "Leave_Policy.pdf",
        "department": "hr",
        "page_label": "2"
    },
    "results": [
        {
            "rank": 1,
            "score": 0.54,
            "text": "...",
            "metadata": {
                "file_name": "Leave_Policy.pdf",
                "page_label": "2"
            }
        }
    ]
}
```

---

# ▶️ Running the Project

## 1. Create Virtual Environment

```bash
python3 -m venv .venv
```

## 2. Activate Virtual Environment

```bash
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Start Ollama

Make sure Ollama is running locally.

Default Ollama endpoint:

```text
http://localhost:11434
```

## 5. Start FastAPI

```bash
python -m uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 🧪 Testing

The RAG pipeline can be tested using:

```bash
python -m app.rag.test_rag
```

Semantic retrieval can be tested using:

```bash
python -m app.retrieval.test_search
```

FastAPI APIs can be tested through Swagger:

```text
http://127.0.0.1:8000/docs
```

---

# 🐳 Docker

The project also contains Docker configuration:

```text
Dockerfile
docker-compose.yml
```

Docker can be used to package and run the application in a consistent environment.

---

# 📚 Supported Document Categories

Current enterprise documents are organized by category:

```text
data/pdf/

├── cloud/
│   ├── DigitalOcean_policies.pdf
│   └── aws_policy.pdf
│
├── hr/
│   └── Leave_Policy.pdf
│
└── travel/
    └── travel_policy.pdf
```

This directory organization also provides a foundation for department/category-based metadata filtering.

---

# 🎯 Project Goals

The main goal is to build a practical Enterprise RAG system demonstrating core AI Engineering concepts:

* Document ingestion
* PDF processing
* Document chunking
* Embedding generation
* Vector indexing
* FAISS semantic search
* Metadata management
* Metadata filtering
* Local LLM integration
* Prompt construction
* Retrieval-Augmented Generation
* Source attribution
* FastAPI REST APIs
* Docker-based deployment
* Modular software architecture

---

# 🧠 Key Concepts Implemented

```text
Document Processing
        +
Chunking
        +
Embeddings
        +
Vector Search
        +
FAISS
        +
Semantic Retrieval
        +
Metadata Filtering
        +
Prompt Engineering
        +
LLM Generation
        +
Source Attribution
        +
FastAPI
        +
Docker
```

---

# 📈 Development Progress

```text
                    Enterprise RAG

                         │
                         ▼
                  PDF Documents
                         │
                         ▼
                 LlamaIndex Loader
                         │
                         ▼
                     Chunking
                         │
                         ▼
                    Embeddings
                         │
                         ▼
                       FAISS
                         │
                         ▼
                  Semantic Search
                         │
                         ▼
                Metadata Filtering
                         │
                         ▼
                    RAG Pipeline
                         │
                         ▼
                    Ollama LLM
                         │
                         ▼
                    FastAPI API
                         │
                         ▼
                   Answer + Sources
```

### Milestone Completion

```text
Milestone 1  — Project Setup          ✅
Milestone 2  — PDF Loading            ✅
Milestone 3  — Document Chunking      ✅
Milestone 4  — Embeddings             ✅
Milestone 5  — FAISS Indexing         ✅
Milestone 6  — Semantic Search        ✅
Milestone 7  — Ollama LLM             ✅
Milestone 8  — RAG Pipeline           ✅
Milestone 9  — FastAPI APIs           ✅
Milestone 10 — Metadata Filtering     ✅
```

# 🎉 Current Status

**Enterprise RAG — Milestones 1–10 Complete**

The project currently provides a complete modular RAG pipeline with:

```text
PDF Ingestion
      ↓
Chunking
      ↓
Embeddings
      ↓
FAISS
      ↓
Semantic Search
      ↓
Metadata Filtering
      ↓
Ollama LLM
      ↓
RAG
      ↓
FastAPI
```

The foundation is now ready for the next stage of Enterprise RAG development, including advanced retrieval, evaluation, reranking, hybrid search, observability, security, and production deployment.
