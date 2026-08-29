# Enterprise RAG

An end-to-end Enterprise Retrieval-Augmented Generation (RAG) system built with **Python, LlamaIndex, FAISS CPU, BM25, Ollama, and FastAPI**.

The system processes enterprise PDF documents, creates semantic chunks, generates embeddings using Ollama, stores vectors in FAISS, performs semantic and keyword retrieval, combines both retrieval strategies using **Reciprocal Rank Fusion (RRF)**, applies metadata filtering, and uses a local Ollama LLM to generate grounded answers.

The project is designed as a modular foundation for building production-oriented Enterprise RAG systems.

---

# 🚀 Project Status

| Milestone | Description                        | Status     |
| --------- | ---------------------------------- | ---------- |
| 1         | Project Setup                      | ✅ Complete |
| 2         | PDF Document Loading               | ✅ Complete |
| 3         | Document Chunking                  | ✅ Complete |
| 4         | Embedding Generation               | ✅ Complete |
| 5         | FAISS Vector Indexing              | ✅ Complete |
| 6         | Semantic Search                    | ✅ Complete |
| 7         | Ollama LLM Integration             | ✅ Complete |
| 8         | Complete RAG Pipeline              | ✅ Complete |
| 9         | FastAPI APIs                       | ✅ Complete |
| 10        | Metadata Filtering                 | ✅ Complete |
| 11        | Hybrid Search — FAISS + BM25 + RRF | ✅ Complete |
| 12        | Reranking                          | 🔜 Next    |

### Current Progress

**11/11 implemented milestones completed 🎉**

**Next milestone: Reranking**

---

# 🏗️ Architecture

```text
                         ┌─────────────────────────┐
                         │   Enterprise Documents  │
                         │        PDF Files        │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │    LlamaIndex Loader    │
                         │      PDF + Metadata     │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │        Chunking         │
                         │    SentenceSplitter     │
                         │   chunk=256 overlap=30  │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   Embedding Generation  │
                         │         Ollama          │
                         │    nomic-embed-text     │
                         │      768 dimensions     │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │    FAISS Vector Store   │
                         │       IndexFlatL2       │
                         │                         │
                         │      index.faiss        │
                         │      metadata.pkl       │
                         └────────────┬────────────┘
                                      │
                                      │
                         ┌────────────▼────────────┐
                         │      User Question      │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │    Query Processing     │
                         └────────────┬────────────┘
                                      │
                         ┌────────────┴────────────┐
                         │                         │
                         ▼                         ▼
              ┌───────────────────┐    ┌───────────────────┐
              │  FAISS Semantic   │    │   BM25 Keyword    │
              │      Search       │    │      Search       │
              └─────────┬─────────┘    └─────────┬─────────┘
                        │                        │
                        │      node_id           │
                        │      matching          │
                        └───────────┬────────────┘
                                    │
                                    ▼
                         ┌─────────────────────────┐
                         │   RRF Hybrid Fusion     │
                         │ Reciprocal Rank Fusion  │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   Hybrid Top-K Results  │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │      Reranking          │
                         │       (Next)             │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │    Context Construction │
                         │            +            │
                         │    Prompt Construction  │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │       Ollama LLM        │
                         │      Qwen / Llama       │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │     Answer + Sources    │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │        FastAPI          │
                         │                         │
                         │  GET  /health           │
                         │  POST /ingest           │
                         │  POST /search            │
                         │  POST /hybrid-search     │
                         │  POST /chat              │
                         └─────────────────────────┘
```

---

# 🔄 RAG Pipeline

The retrieval pipeline now supports both semantic and keyword-based retrieval.

```text
User Question
      │
      ▼
Query Processing
      │
      ├───────────────────────┐
      │                       │
      ▼                       ▼
FAISS Semantic Search       BM25 Search
      │                       │
      │                       │
      └───────────┬───────────┘
                  │
                  ▼
           RRF Hybrid Fusion
                  │
                  ▼
          Hybrid Top-K Results
                  │
                  ▼
              Reranking
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
* BM25
* bm25s
* Ollama
* Nomic Embed Text
* Qwen / Llama

## Data Processing

* LlamaIndex PDF Loader
* LlamaIndex `SentenceSplitter`
* NumPy
* Pickle
* JSON

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
│   │   ├── bm25_search.py
│   │   ├── hybrid_search.py
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
├── bm25_index/
│   ├── doc_ids.txt
│   └── doc_metadata.json
│
├── vectorstore/
│   ├── index.faiss
│   └── metadata.pkl
│
├── docs/
├── logs/
└── tests/
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

Chunk text + metadata + node_id
```

Each document chunk has a unique `node_id`.

The `node_id` is used to identify the same chunk across different retrieval systems.

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
* Node ID
* Text
* Metadata

## FAISS Distance

The current index uses:

```text
IndexFlatL2
```

Therefore the returned score represents **L2 distance**.

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

The retrieved sources provide transparency and allow users to identify where the answer originated.

---

# 🌐 Milestone 9 — FastAPI APIs

The RAG system is exposed through FastAPI REST APIs.

Current API operations include:

```text
GET  /health
POST /ingest
POST /search
POST /hybrid-search
POST /chat
```

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

## Document Ingestion

```text
POST /ingest
```

The ingestion endpoint processes the configured documents and creates/updates the FAISS vector index.

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

## Semantic Search

```text
POST /search
```

The search endpoint performs semantic retrieval and supports metadata filtering.

## Hybrid Search

```text
POST /hybrid-search
```

The hybrid search endpoint combines:

```text
FAISS Semantic Search
        +
BM25 Keyword Search
        ↓
RRF Fusion
        ↓
Hybrid Results
```

---

# 🔐 Milestone 10 — Metadata Filtering

Metadata filtering has been implemented as part of the retrieval layer.

The retrieval system supports:

```text
file_name
department
page_label
```

This allows semantic search and keyword search to be combined with structured document filtering.

## Metadata Filtering Flow

```text
User Question
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

All supplied filters are combined using AND logic.

Example:

```text
file_name = Leave_Policy.pdf

AND

department = HR

AND

page_label = 2
```

A result must satisfy all supplied filters to be returned.

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

before applying the final ranking/filtering process.

This improves retrieval robustness when some high-ranked candidates do not satisfy the requested filters.

---

# 🔀 Milestone 11 — Hybrid Search

Hybrid Search combines **semantic retrieval** and **keyword retrieval** to improve recall.

The project uses:

```text
FAISS
+
BM25
+
Reciprocal Rank Fusion (RRF)
```

## Why Hybrid Search?

Semantic search is useful for understanding the meaning of a query.

BM25 is useful for exact or lexical matches such as:

```text
AWS
Leave Policy
Paternity leave
Docker
API
```

Combining both approaches provides a stronger retrieval strategy.

---

## Hybrid Search Architecture

```text
                 User Question
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
       FAISS Search          BM25 Search
       Semantic              Keyword
             │                   │
             │                   │
             └─────────┬─────────┘
                       │
                       ▼
                Node ID Matching
                       │
                       ▼
                RRF Fusion
                       │
                       ▼
                 Ranked Results
                       │
                       ▼
                    Top-K
```

---

## FAISS Retrieval

FAISS performs semantic retrieval using the question embedding.

```text
Question
   ↓
Ollama Embedding
   ↓
FAISS
   ↓
Semantic Candidates
```

Each result contains a unique:

```text
node_id
```

---

## BM25 Retrieval

BM25 performs lexical keyword retrieval.

Implementation:

```text
app/retrieval/bm25_search.py
```

The BM25 index is stored separately:

```text
bm25_index/

├── doc_ids.txt
├── doc_metadata.json
└── BM25 index files
```

BM25 uses the same chunk `node_id` so that results can be matched with FAISS results.

---

## Node ID Strategy

Every chunk has a unique `node_id`.

Example:

```text
98b6fd39-50f7-420d-bae7-34bfbc2735c8
bc2d05cd-5a4c-4e51-bd7e-dd50e647ba6c
cafca7ee-4900-43e8-a5a0-009751e307d6
075500c7-53df-44d2-b594-2ed8d70173ce
6cccf679-3a96-4e83-9ff1-1ff849c58a1c
```

The same `node_id` can appear in both FAISS and BM25 results when both systems retrieve the same chunk.

This allows the hybrid layer to identify that both retrieval systems refer to the same document chunk.

---

# 🔢 Reciprocal Rank Fusion

The hybrid search uses **Reciprocal Rank Fusion (RRF)** to combine rankings.

The scoring formula is:

```text
RRF Score = 1 / (k + rank)
```

The implementation uses:

```python
RRF_K = 60
```

If a chunk appears in both FAISS and BM25:

```text
FAISS rank contribution
        +
BM25 rank contribution
        ↓
Combined RRF score
```

The final results are sorted by the combined RRF score.

---

## Hybrid Search Example

Question:

```text
What are the types of leave in Goa?
```

The system performs:

```text
                    Question
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
       FAISS                      BM25
          │                         │
     node_id=A                node_id=A
     node_id=B                node_id=C
     node_id=C                node_id=D
          │                         │
          └────────────┬────────────┘
                       ▼
                    RRF
                       │
                       ▼
               Combined Ranking
                       │
                       ▼
                    Top 5
```

If FAISS and BM25 both return the same `node_id`, the result is treated as **one chunk** with a combined RRF score.

---

# 📡 Hybrid Search API

Endpoint:

```text
POST /hybrid-search
```

Example request:

```json
{
    "question": "What are the types of leave in Goa?",
    "top_k": 5
}
```

The endpoint returns:

```json
{
    "question": "What are the types of leave in Goa?",
    "search_type": "hybrid",
    "results": [
        {
            "rank": 1,
            "node_id": "cafca7ee-4900-43e8-a5a0-009751e307d6",
            "hybrid_score": 0.016393,
            "text": "...",
            "metadata": {
                "file_name": "Leave_Policy.pdf",
                "page_label": "1"
            }
        }
    ]
}
```

---

# 🎯 Project Goals

The main goal is to build a practical Enterprise RAG system demonstrating core AI Engineering concepts:

* Document ingestion
* PDF processing
* Document chunking
* Embedding generation
* Vector indexing
* FAISS semantic search
* BM25 keyword search
* Hybrid retrieval
* Reciprocal Rank Fusion
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
BM25
        +
Hybrid Search
        +
RRF Fusion
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
                    BM25 Search
                         │
                         ▼
                  Hybrid Retrieval
                         │
                         ▼
                    RRF Fusion
                         │
                         ▼
                     Reranking
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

---

# 🏁 Milestone Completion

```text
Milestone 1  — Project Setup              ✅
Milestone 2  — PDF Loading                ✅
Milestone 3  — Document Chunking          ✅
Milestone 4  — Embeddings                 ✅
Milestone 5  — FAISS Indexing             ✅
Milestone 6  — Semantic Search            ✅
Milestone 7  — Ollama LLM                 ✅
Milestone 8  — RAG Pipeline               ✅
Milestone 9  — FastAPI APIs               ✅
Milestone 10 — Metadata Filtering          ✅
Milestone 11 — Hybrid Search + RRF         ✅
Milestone 12 — Reranking                  🔜
```

---

# 🎉 Current Status

**Enterprise RAG — Milestones 1–11 Complete**

The project currently provides:

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
BM25 Keyword Search
      ↓
Hybrid Search
      ↓
RRF Fusion
      ↓
Metadata Filtering
      ↓
Ollama LLM
      ↓
RAG
      ↓
FastAPI
      ↓
Answer + Sources
```

The retrieval foundation is now ready for the next stage of Enterprise RAG development.

## Next Stage

The next milestone is **Reranking**.

The planned retrieval architecture is:

```text
User Query
    ↓
FAISS + BM25
    ↓
Hybrid Retrieval
    ↓
RRF Fusion
    ↓
Candidate Top-K
    ↓
Reranker
    ↓
Best Relevant Chunks
    ↓
Context
    ↓
LLM
    ↓
Grounded Answer
```

Future improvements can include:

* Reranking
* Retrieval evaluation
* RAG evaluation metrics
* Observability
* Query rewriting
* Access control
* Security
* Caching
* Production deployment
* Cloud infrastructure
* Monitoring

```
```
