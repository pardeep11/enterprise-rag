# Enterprise RAG

An end-to-end Retrieval-Augmented Generation (RAG) application built with Python, LlamaIndex, FAISS CPU, Ollama, and FastAPI.

The project loads PDF documents, splits them into chunks, generates embeddings, stores vectors in FAISS, performs semantic search, and uses an Ollama LLM to generate answers from retrieved context.

---

## 🚀 Project Status

| Milestone | Description | Status |
|---|---|---|
| 1 | Project Setup | ✅ Complete |
| 2 | PDF Document Loading | ✅ Complete |
| 3 | Document Chunking | ✅ Complete |
| 4 | Embedding Generation | ✅ Complete |
| 5 | FAISS Vector Indexing | ✅ Complete |
| 6 | Semantic Search | ✅ Complete |
| 7 | Ollama LLM Integration | ✅ Complete |
| 8 | Complete RAG Pipeline | ✅ Complete |
| 9 | FastAPI APIs | ✅ Complete |
| 10 | Metadata Filtering | 🔜 Next |

---

## 🏗️ Architecture

```text
PDF Documents
      ↓
LlamaIndex PDF Loader
      ↓
Document Chunking
      ↓
Ollama Embeddings
      ↓
FAISS CPU
      ↓
Semantic Search
      ↓
Top 5 Relevant Chunks
      ↓
Chat Prompt
      ↓
Ollama LLM
      ↓
Final Answer
      ↓
FastAPI
````

---

## 🔄 RAG Pipeline

The complete RAG workflow:

```text
Question
   ↓
Generate Question Embedding
   ↓
FAISS Similarity Search
   ↓
Retrieve Top 5 Chunks
   ↓
Build Prompt With Context
   ↓
Send Prompt To Ollama LLM
   ↓
Generate Answer
```
---

## 🧰 Tech Stack

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

The project is organized into separate modules for:

* PDF loading
* Document chunking
* Embeddings
* Vector storage
* RAG pipeline
* FastAPI

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

The purpose of chunking is to create smaller pieces of text that can be embedded and searched efficiently.

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

Example:

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

The generated embeddings are converted into a NumPy matrix and stored in FAISS.

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

The metadata file stores the original chunk text and document metadata so that retrieved vectors can be mapped back to their source content.

---

# 🔎 Milestone 6 — Semantic Search

Semantic search retrieves the most relevant chunks for a user's question.

Flow:

```text
Question
   ↓
Question Embedding
   ↓
FAISS Similarity Search
   ↓
Top 5 Relevant Chunks
```

Example question:

```text
What is the maximum casual leave allowed per month?
```

The system retrieves relevant sections from:

```text
Leave_Policy.pdf
```

The retrieved results include:

* Rank
* Similarity score
* Text
* Document metadata

---

# 🤖 Milestone 7 — Ollama LLM

Ollama is used to run the LLM locally.

The application connects to a locally running Ollama server.

Example:

```text
Application
     ↓
Ollama
     ↓
Local LLM
     ↓
Response
```

This allows the RAG application to use a locally running model instead of depending on a hosted LLM API.

---

# 🧩 Milestone 8 — Complete RAG Pipeline

The complete RAG pipeline combines retrieval and generation.

```text
User Question
      ↓
Question Embedding
      ↓
FAISS Similarity Search
      ↓
Top 5 Relevant Chunks
      ↓
Chat Prompt
      ↓
Ollama LLM
      ↓
Generated Answer
```

The application also returns the retrieved sources.

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

# 🌐 Current API

The current RAG application exposes a question-answering endpoint.

## POST `/ask`

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
    "sources": [...]
}
```

The `sources` field contains the chunks retrieved by FAISS and their document metadata.

---

# ▶️ Running the Project

## 1. Create virtual environment

```bash
python3 -m venv .venv
```

## 2. Activate virtual environment

```bash
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Start Ollama

Make sure Ollama is running locally.

The application uses:

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

The RAG pipeline can be tested from the terminal.

```bash
python -m app.rag.test_rag
```

The FastAPI `/ask` endpoint can be tested using:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
-H "Content-Type: application/json" \
-d '{"question":"What is the maximum casual leave allowed per month?"}'
```

---

# 🔜 Upcoming Milestones

## Milestone 9 — FastAPI APIs

Planned APIs:

```text
GET  /health
POST /ingest
POST /search
POST /chat
```

---

## Milestone 10 — Metadata Filtering

The RAG system will support metadata-based filtering.

Planned metadata:

```text
filename
page
department
policy_type
```

Example filters:

```text
Search only HR
Search only Travel
Search only Cloud
```

---

# 🎯 Project Goal

The goal of this project is to build a practical, modular RAG system that demonstrates:

* Document ingestion
* Document chunking
* Embedding generation
* Vector indexing
* Semantic search
* Metadata handling
* Local LLM integration
* Retrieval-Augmented Generation
* REST APIs

The project is being developed incrementally through milestones to understand each component of a production-style RAG architecture.

---

# 📈 Development Progress

```text
PDF
 ↓
LlamaIndex
 ↓
Chunking
 ↓
Embeddings
 ↓
FAISS
 ↓
Semantic Search
 ↓
Ollama
 ↓
RAG
 ↓
FastAPI
```

Current progress:

```text
Milestones 1–8 completed ✅

Milestones 9–10 next 🔜
```

---

# 👨‍💻 Project Development

This project is being developed milestone-by-milestone to build practical understanding of:

```text
RAG
+
Vector Search
+
LLM
+
FastAPI
+
Local AI
```