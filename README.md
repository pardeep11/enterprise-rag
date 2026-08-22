# Enterprise RAG

An end-to-end Retrieval-Augmented Generation (RAG) application built with Python, LlamaIndex, FAISS CPU, Ollama, and FastAPI.

The project loads PDF documents, splits them into chunks, generates embeddings, stores vectors in FAISS, performs semantic search, applies metadata filtering, and uses an Ollama LLM to generate answers from retrieved context.

---

## 🚀 Project Status

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

**Current Progress: 10/10 milestones completed 🎉**

---

# 🏗️ Architecture

```text
PDF Documents
      ↓
LlamaIndex PDF Loader
      ↓
Document Chunking
      ↓
Ollama Embeddings
      ↓
FAISS Vector Index
      ↓
Semantic Search
      ↓
Candidate Retrieval
      ↓
Metadata Filtering
      ↓
Top Relevant Chunks
      ↓
Prompt Construction
      ↓
Ollama LLM
      ↓
Generated Answer
      ↓
FastAPI
```

---

# 🔄 RAG Pipeline

The complete RAG workflow:

```text
User Question
      ↓
Generate Question Embedding
      ↓
FAISS Vector Search
      ↓
Retrieve Candidate Chunks
      ↓
Apply Metadata Filters
      ↓
Select Top-K Relevant Chunks
      ↓
Build Prompt With Context
      ↓
Send Prompt To Ollama LLM
      ↓
Generate Answer
      ↓
Return Answer + Sources
```

---

# 🧰 Tech Stack

* Python
* FastAPI
* LlamaIndex
* FAISS CPU
* Ollama
* Nomic Embed Text
* Qwen / Llama
* NumPy
* PyMuPDF
* Pydantic
* Uvicorn
* Pickle

---

# 📂 Project Structure

```text
enterprise-rag/
│
├── app/
│   ├── loaders/
│   │   └── pdf_loader.py
│   │
│   ├── chunking/
│   │   └── chunker.py
│   │
│   ├── embeddings/
│   │   └── embedder.py
│   │
│   ├── vectorstore/
│   │   └── index.py
│   │
│   └── rag/
│       ├── rag_pipeline.py
│       └── test_rag.py
│
├── data/
│   └── pdf/
│       ├── hr/
│       │   └── Leave_Policy.pdf
│       │
│       └── cloud/
│           └── aws_policy.pdf
│
├── vectorstore/
│   ├── index.faiss
│   └── metadata.pkl
│
├── main.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

# 📌 Milestone 1 — Project Setup

Created the initial Enterprise RAG project structure.

The application is organized into modular components for:

* PDF loading
* Document chunking
* Embedding generation
* Vector storage
* Semantic search
* RAG pipeline
* FastAPI APIs

The modular structure allows each component to be developed, tested, and extended independently.

---

# 📄 Milestone 2 — PDF Loading

PDF documents are loaded using LlamaIndex.

Current test documents include:

```text
Leave_Policy.pdf
aws_policy.pdf
```

The loader extracts document content and metadata such as:

```text
file_name
file_path
file_type
page_label
file_size
creation_date
last_modified_date
```

This metadata is later used during metadata-based retrieval.

---

# ✂️ Milestone 3 — Document Chunking

Documents are split into smaller chunks using LlamaIndex `SentenceSplitter`.

Current configuration:

```python
SentenceSplitter(
    chunk_size=256,
    chunk_overlap=30,
)
```

Chunking allows large documents to be divided into smaller semantic units that can be efficiently embedded and retrieved.

---

# 🧠 Milestone 4 — Embedding Generation

The project uses Ollama locally with:

```text
nomic-embed-text
```

Each document chunk is converted into a numerical vector.

Current embedding dimension:

```text
768
```

Embedding workflow:

```text
Text Chunk
    ↓
Ollama
    ↓
nomic-embed-text
    ↓
768-dimensional vector
```

---

# 🗄️ Milestone 5 — FAISS Vector Index

FAISS CPU is used for vector similarity search.

The generated embeddings are converted into a NumPy matrix and stored in a FAISS index.

Output files:

```text
vectorstore/

├── index.faiss
└── metadata.pkl
```

Current index information:

```text
FAISS vectors : 91
Dimensions    : 768
```

The FAISS index stores the vectors, while `metadata.pkl` stores the corresponding chunk text and document metadata.

This allows a retrieved FAISS vector ID to be mapped back to the original document chunk.

---

# 🔎 Milestone 6 — Semantic Search

Semantic search retrieves document chunks that are semantically relevant to a user's question.

Flow:

```text
Question
   ↓
Question Embedding
   ↓
FAISS Vector Search
   ↓
Relevant Chunks
```

Example question:

```text
What is the maximum casual leave allowed per month?
```

The system converts the question into an embedding and searches the FAISS index for the closest vectors.

The retrieved results include:

* Rank
* Distance score
* Text
* Document metadata

> Note: The current FAISS index uses L2 distance, so a lower distance generally represents a more similar vector. The distance value should not be interpreted as a percentage similarity score.

---

# 🤖 Milestone 7 — Ollama LLM Integration

Ollama is used to run the LLM locally.

The application connects to the locally running Ollama server.

Architecture:

```text
Application
     ↓
Ollama
     ↓
Local LLM
     ↓
Generated Response
```

Using Ollama allows the project to run the LLM locally without depending entirely on a hosted LLM API.

Models such as Qwen / Llama can be used for response generation.

---

# 🧩 Milestone 8 — Complete RAG Pipeline

The complete RAG pipeline combines retrieval and generation.

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
Generated Answer
      ↓
Sources
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

The `sources` field provides transparency by showing the document chunks used during retrieval.

---

# 🌐 Milestone 9 — FastAPI APIs

The RAG application has been exposed through FastAPI REST APIs.

The API layer provides an interface between clients and the RAG pipeline.

Core API operations include:

```text
GET  /health
POST /ingest
POST /search
POST /ask
```

### Health Check

```text
GET /health
```

Used to verify that the API service is running.

Example response:

```json
{
    "status": "healthy"
}
```

### Document Ingestion

```text
POST /ingest
```

Triggers document ingestion and builds the FAISS vector index.

### Semantic Search

```text
POST /search
```

Used to search the indexed documents and return relevant chunks.

### RAG Question Answering

```text
POST /ask
```

Used to send a question to the RAG pipeline and receive:

* Generated answer
* Retrieved sources
* Document metadata

Example request:

```json
{
    "question": "What is the maximum casual leave allowed per month?"
}
```

---

# 🔐 Milestone 10 — Metadata Filtering

Metadata filtering has now been implemented.

The search system can combine:

**Semantic Search + Metadata Filtering**

Supported filters include:

```text
file_name
department
page_label
```

Example:

```text
Question:
What is earned leave?

Filters:

file_name  = Leave_Policy.pdf
department = hr
page_label = 2
```

The system first performs vector search and retrieves multiple candidates.

It then applies the metadata filters before returning the final results.

---

## Metadata Filtering Flow

```text
User Question
      ↓
Question Embedding
      ↓
FAISS Search
      ↓
Retrieve Candidate Results
      ↓
Metadata Filtering
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

## Candidate Retrieval Strategy

The system retrieves more candidates than the requested `top_k` before applying metadata filters.

For example:

```python
candidate_k = top_k * 4
```

If:

```text
top_k = 5
```

the system initially retrieves:

```text
5 × 4 = 20 candidates
```

Then metadata filtering is applied.

```text
FAISS
  ↓
20 candidates
  ↓
Metadata filtering
  ↓
Valid results
  ↓
Top 5
```

This approach helps prevent relevant results from being lost when some of the highest-ranked semantic results do not satisfy the requested metadata filters.

---

# 🔍 Metadata Matching Logic

The current implementation supports:

### File Name

```text
Leave_Policy.pdf
```

Only chunks belonging to the requested file are returned.

### Department

The department is determined from the document path.

Example:

```text
documents/hr/Leave_Policy.pdf
```

The system can identify:

```text
department = hr
```

### Page

The search can be restricted to a specific page:

```text
page_label = 2
```

All supplied filters are treated as AND conditions.

For example:

```text
file_name = Leave_Policy.pdf
AND
department = hr
AND
page_label = 2
```

A chunk must satisfy all requested filters to be included in the final results.

---

# 📡 Current RAG API Example

### Request

```json
{
    "question": "What is the maximum casual leave allowed per month?"
}
```

### Response

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
python -m uvicorn main:app --reload
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

The RAG pipeline can be tested from the terminal:

```bash
python -m app.rag.test_rag
```

The FastAPI `/ask` endpoint can be tested using:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
-H "Content-Type: application/json" \
-d '{"question":"What is the maximum casual leave allowed per month?"}'
```

The `/search` endpoint can be used to test semantic search and metadata filtering independently.

---

# 🎯 Project Goal

The goal of this project is to build a practical, modular, production-style RAG system demonstrating:

* Document ingestion
* PDF processing
* Document chunking
* Embedding generation
* Vector indexing
* FAISS similarity search
* Metadata management
* Metadata filtering
* Local LLM integration
* Retrieval-Augmented Generation
* Source attribution
* REST APIs
* FastAPI
* Local AI infrastructure

The project is developed incrementally through milestones to understand the architecture and implementation of enterprise-grade RAG systems.

---

# 📈 Development Progress

```text
PDF Documents
      ↓
LlamaIndex
      ↓
Document Chunking
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
RAG Pipeline
      ↓
FastAPI
```

### Current Progress

```text
Milestone 1  — Project Setup          ✅
Milestone 2  — PDF Loading            ✅
Milestone 3  — Chunking               ✅
Milestone 4  — Embeddings             ✅
Milestone 5  — FAISS Indexing         ✅
Milestone 6  — Semantic Search        ✅
Milestone 7  — Ollama LLM             ✅
Milestone 8  — RAG Pipeline           ✅
Milestone 9  — FastAPI APIs           ✅
Milestone 10 — Metadata Filtering     ✅
```

**10/10 Milestones Completed 🎉**

---

# 🧠 Key RAG Concepts Implemented

This project currently demonstrates the following important AI Engineering concepts:

```text
Document Processing
        +
Chunking
        +
Embeddings
        +
Vector Database / FAISS
        +
Semantic Retrieval
        +
Metadata Filtering
        +
Prompt Construction
        +
LLM Generation
        +
Source Attribution
        +
FastAPI
```

The next phase can build on this foundation with more advanced Enterprise-RAG capabilities such as **hybrid search, reranking, evaluation, conversation memory, authentication, observability, and production deployment**.
