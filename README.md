# Enterprise RAG

Enterprise-oriented Retrieval-Augmented Generation (RAG) system for querying internal documents using hybrid retrieval, reranking, metadata filtering, and local LLM inference.

## Architecture

```text
PDF Documents
     ↓
Document Loading
     ↓
Chunking
     ↓
Embeddings
     ↓
FAISS + BM25
     ↓
Hybrid Retrieval + RRF
     ↓
Candidate Retrieval
     ↓
Cross-Encoder Reranking
     ↓
Top-K Context
     ↓
Ollama LLM
     ↓
Answer + Sources
```

## Tech Stack

* **Python**
* **FastAPI**
* **LlamaIndex**
* **FAISS**
* **BM25**
* **Ollama**
* **nomic-embed-text**
* **Cross-Encoder**
* **Docker**

## Retrieval

### Semantic Search

Uses `nomic-embed-text` embeddings with FAISS for semantic similarity search.

### BM25

Keyword-based retrieval for exact terms and enterprise-specific terminology.

### Hybrid Retrieval

Combines semantic and BM25 retrieval using Reciprocal Rank Fusion (RRF).

### Reranking

Hybrid retrieval generates a larger candidate set, which is then reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

For the current configuration:

```text
Final Top-K:     3
Candidate Pool:  9
```

### MMR

MMR retrieval is also evaluated to compare relevance and diversity.

## Chunking

```text
Chunk size:    256
Chunk overlap: 30
```

Sentence-based chunking is used to preserve context across chunk boundaries.

## Metadata Filtering

Chunks contain metadata such as:

```text
filename
page
department
policy_type
```

This supports filtered retrieval based on document attributes.

## API

| Method | Endpoint  | Purpose                  |
| ------ | --------- | ------------------------ |
| GET    | `/health` | Health check             |
| POST   | `/ingest` | Document ingestion       |
| POST   | `/search` | Retrieve relevant chunks |
| POST   | `/chat`   | RAG question answering   |

## Retrieval Evaluation

Evaluation dataset:

```text
Questions: 20
Top-K:     3
```

### Results

| Method                 |   Hit@3 |        MRR |
| ---------------------- | ------: | ---------: |
| Hybrid Search          |     80% |     0.6667 |
| **Hybrid + Reranking** | **90%** |     0.7083 |
| MMR                    |     85% | **0.7167** |

### Result

* **Hybrid + Reranking** achieved the highest Hit@3 at **90%**.
* **MMR** achieved the highest MRR at **0.7167**.

## Failure Analysis

For the two failed Hybrid + Reranking questions, the expected ground-truth chunks were not present in the initial 9 Hybrid candidates.

Therefore, these failures occurred during **initial retrieval**, not reranking.

```text
Query
  ↓
Hybrid Retrieval
  ↓
Expected chunk not retrieved
  ↓
Reranker cannot recover it
```

## Evaluation Report

The evaluation generates:

```text
evaluation_report.json
```

containing the benchmark results and improvement metrics.

## Project Status

* [x] PDF ingestion
* [x] Chunking
* [x] Embeddings
* [x] FAISS semantic search
* [x] BM25 search
* [x] Hybrid retrieval
* [x] RRF
* [x] Metadata filtering
* [x] MMR
* [x] Cross-Encoder reranking
* [x] FastAPI APIs
* [x] Retrieval evaluation
* [x] Failure analysis

