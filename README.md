# Enterprise RAG

An end-to-end Retrieval-Augmented Generation (RAG) application built with Python, LlamaIndex, FAISS CPU, Ollama, and FastAPI.

The project loads PDF documents, splits them into chunks, generates embeddings, stores vectors in FAISS, performs semantic search, and uses an Ollama LLM to generate answers from the retrieved context.

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
| 9 | FastAPI APIs | 🔜 Next |
| 10 | Metadata Filtering | 🔜 Next |

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