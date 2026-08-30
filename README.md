# Enterprise RAG

An end-to-end Enterprise Retrieval-Augmented Generation (RAG) system built with **Python, LlamaIndex, FAISS CPU, BM25, Ollama, Sentence Transformers, and FastAPI**.

The system processes enterprise PDF documents, creates semantic chunks, generates embeddings using Ollama, stores vectors in FAISS, performs semantic and keyword retrieval, combines both retrieval strategies using **Reciprocal Rank Fusion (RRF)**, applies metadata filtering, reranks retrieved candidates using a **Cross-Encoder**, and uses a local Ollama LLM to generate grounded answers.

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
| 12        | Reranking                          | ✅ Complete |

### Current Progress

**12/12 implemented milestones completed 🎉**

The retrieval pipeline now supports:

```text
FAISS Semantic Search
        +
BM25 Keyword Search
        ↓
RRF Hybrid Fusion
        ↓
Candidate Retrieval
        ↓
Cross-Encoder Reranking
        ↓
Final Top-K Results
```

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
                   ┌─────────────┴─────────────┐
                   │                           │
                   ▼                           ▼
          ┌─────────────────┐        ┌─────────────────┐
          │ FAISS Semantic  │        │  BM25 Keyword   │
          │     Search      │        │     Search      │
          └────────┬────────┘        └────────┬────────┘
                   │                           │
                   └─────────────┬─────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      RRF Fusion         │
                    │ Reciprocal Rank Fusion  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Candidate Documents   │
                    │     e.g. Top 15         │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │       Reranking         │
                    │      Cross-Encoder      │
                    │ MS MARCO MiniLM L-6-v2  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    Final Top-K Results  │
                    │       e.g. Top 5        │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Context Construction  │
                    │           +             │
                    │   Prompt Construction   │
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
                    └─────────────────────────┘
```

---

# 🔄 RAG Retrieval Pipeline

The retrieval pipeline uses a multi-stage retrieval architecture:

```text
User Question
      │
      ▼
FAISS + BM25
      │
      ▼
Hybrid Retrieval
      │
      ▼
RRF Fusion
      │
      ▼
Candidate Pool
      │
      │  Example:
      │  top_k = 5
      │  candidates = 15
      ▼
Cross-Encoder Reranker
      │
      ▼
Final Top-K
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

The important distinction is:

```text
Retriever
    ↓
Find candidate documents

Reranker
    ↓
Reorder candidate documents by relevance
```

The reranker does **not** search the entire document collection again. It takes the candidate documents returned by the retrieval stage and produces a better relevance ordering.

---

# 🎯 Milestone 12 — Reranking

Reranking has been implemented using a **Cross-Encoder** from Sentence Transformers.

Implementation:

```text
app/retrieval/reranking.py
```

Model:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

## Why Reranking?

Initial retrieval methods such as FAISS and BM25 are optimized for efficiently finding candidate documents.

However, the first ranking may not always place the most relevant document at the top.

For example:

```text
Initial Hybrid Ranking

Rank 1 → Chunk A
Rank 2 → Chunk B
Rank 3 → Chunk C
Rank 4 → Chunk D
Rank 5 → Chunk E
Rank 6 → Chunk F
Rank 7 → Chunk G ← highly relevant
Rank 8 → Chunk H
Rank 9 → Chunk I ← highly relevant
Rank 10 → Chunk J
```

The reranker evaluates the query together with each candidate document:

```text
Query + Candidate Document
          ↓
     Cross-Encoder
          ↓
   Relevance Score
```

It then sorts the candidates by the Cross-Encoder score:

```text
Before Reranking

7 → Highly Relevant
9 → Highly Relevant
1
2
3
...

        ↓ Reranker

After Reranking

1 → Highly Relevant
2 → Highly Relevant
3
4
5
```

Therefore, the most relevant chunks are moved toward the top of the final result list.

---

# 🔢 Candidate Retrieval and Reranking

The system retrieves more candidates than the final requested `top_k`.

For example:

```text
Requested top_k = 5

Candidate pool = top_k × 3

                 ↓

Candidate count = 15
```

The pipeline becomes:

```text
Hybrid Search
      ↓
15 candidate chunks
      ↓
Cross-Encoder
      ↓
Relevance scoring
      ↓
Sort descending
      ↓
Top 5 final chunks
```

This gives the reranker a larger candidate pool from which to select the most relevant results.

---

# 🧠 Cross-Encoder Reranking

The reranker receives pairs containing:

```text
(query, document)
```

Example:

```text
Query:
What are the types of leave in Goa?

Document:
Leave Policy
Casual Leave
Sick Leave
Privilege Leave
...
```

The Cross-Encoder jointly evaluates the query and document and produces a relevance score.

The implementation:

```python
pairs = [
    [query, doc.get("text", "")]
    for doc in documents
]

scores = reranker_model.predict(pairs)
```

The score is attached to each document:

```python
doc["rerank_score"] = float(score)
```

The documents are then sorted in descending order:

```python
sorted_docs = sorted(
    documents,
    key=lambda x: x["rerank_score"],
    reverse=True
)
```

Finally, only the requested number of results are returned:

```python
return sorted_docs[:top_k]
```

---

# 📊 Reranking Example

Example request:

```text
Question:
What are the types of leave in Goa?

top_k:
5
```

The system first retrieves:

```text
15 hybrid candidates
```

The Cross-Encoder then scores all 15 candidates.

Example:

```text
Candidate 1 →  6.55
Candidate 2 →  6.55
Candidate 3 → -1.33
Candidate 4 → -1.33
Candidate 5 → -6.03
...
```

The results are sorted from highest to lowest score.

Example output:

```text
Candidate Pool
      ↓
15 documents
      ↓
Cross-Encoder
      ↓
15 relevance scores
      ↓
Sort descending
      ↓
Top 5
```

A negative reranking score is valid. The Cross-Encoder score is a model output/logit and should **not** be interpreted as a percentage.

The important property is the **relative ordering**:

```text
6.55 > -1.33 > -6.03
```

Therefore:

```text
Higher Cross-Encoder score
        ↓
More relevant

Lower Cross-Encoder score
        ↓
Less relevant
```

---

# 📡 Hybrid Search API

Endpoint:

```text
POST /hybrid-search
```

This endpoint performs hybrid retrieval without reranking.

Pipeline:

```text
FAISS
 +
BM25
 ↓
RRF
 ↓
Top-K
```

Example request:

```json
{
    "question": "What are the types of leave in Goa?",
    "top_k": 5
}
```

---

# 📡 Hybrid Search + Reranking API

Endpoint:

```text
POST /hybrid-rerank-search
```

This endpoint performs hybrid retrieval followed by Cross-Encoder reranking.

Pipeline:

```text
Question
   ↓
FAISS + BM25
   ↓
RRF
   ↓
Candidate Pool
   ↓
Cross-Encoder
   ↓
Final Top-K
```

Example request:

```json
{
    "question": "What are the types of leave in Goa?",
    "top_k": 5,
    "enable_rerank": true
}
```

Example response structure:

```json
{
    "question": "What are the types of leave in Goa?",
    "search_type": "hybrid_with_rerank",
    "candidate_count": 15,
    "result_count": 5,
    "execution_time_seconds": 2.09,
    "timing_breakdown": {
        "retrieval_seconds": 1.07,
        "rerank_seconds": 1.02
    },
    "results": [
        {
            "rank": 1,
            "node_id": "...",
            "hybrid_score": 0.016393,
            "rerank_score": 6.554742,
            "text": "...",
            "metadata": {}
        }
    ]
}
```

The timing breakdown allows the retrieval and reranking latency to be measured separately.

---

# 📂 Project Structure

```text
enterprise-rag/

├── Dockerfile
├── README.md
├── requirements.txt
├── docker-compose.yml
│
├── app/
│   ├── main.py
│   │
│   ├── chunking/
│   │   └── chunker.py
│   │
│   ├── embeddings/
│   │   └── embedder.py
│   │
│   ├── llm/
│   │   └── ollama_llm.py
│   │
│   ├── loaders/
│   │   └── pdf_loader.py
│   │
│   ├── rag/
│   │   ├── rag_pipeline.py
│   │   └── test_rag.py
│   │
│   ├── retrieval/
│   │   ├── search.py
│   │   ├── bm25_search.py
│   │   ├── hybrid_search.py
│   │   ├── reranking.py
│   │   └── test_search.py
│   │
│   └── vectorstore/
│       └── index.py
│
├── data/
│   ├── pdf/
│   └── processed/
│
├── bm25_index/
│
├── vectorstore/
│
├── docs/
├── logs/
└── tests/
```

---

# 🏁 Milestone Completion

```text
Milestone 1  — Project Setup                  ✅
Milestone 2  — PDF Loading                    ✅
Milestone 3  — Document Chunking              ✅
Milestone 4  — Embeddings                     ✅
Milestone 5  — FAISS Indexing                 ✅
Milestone 6  — Semantic Search                ✅
Milestone 7  — Ollama LLM                     ✅
Milestone 8  — RAG Pipeline                   ✅
Milestone 9  — FastAPI APIs                   ✅
Milestone 10 — Metadata Filtering             ✅
Milestone 11 — Hybrid Search + RRF            ✅
Milestone 12 — Reranking                     ✅
```

---

# 🎉 Current Status

**Enterprise RAG — Milestones 1–12 Complete**

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
Candidate Retrieval
      ↓
Cross-Encoder Reranking
      ↓
Final Top-K Results
      ↓
Context Construction
      ↓
Ollama LLM
      ↓
RAG
      ↓
FastAPI
      ↓
Answer + Sources
```

The retrieval pipeline now contains both **multi-retriever fusion** and **neural reranking**, providing a stronger retrieval foundation for the RAG generation stage.

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
* Metadata filtering
* Cross-Encoder reranking
* Candidate retrieval
* Relevance scoring
* Prompt construction
* Local LLM integration
* Retrieval-Augmented Generation
* Source attribution
* FastAPI REST APIs
* Docker-based deployment
* Modular software architecture

---

# 🔮 Future Improvements

After completing the core retrieval pipeline, future improvements can include:

* Retrieval evaluation
* RAG evaluation metrics
* Precision@K / Recall@K
* MRR / NDCG
* Reranker benchmarking
* Query rewriting
* Query expansion
* Context compression
* Observability
* Access control
* Security
* Caching
* Production deployment
* Cloud infrastructure
* Monitoring

````