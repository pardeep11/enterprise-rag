from fastapi import FastAPI
from pydantic import BaseModel, Field
import time

from app.loaders.pdf_loader import load_pdf_documents
from app.chunking.chunker import chunk_documents
from app.vectorstore.index import build_index
from app.retrieval.search import semantic_search
from app.rag.rag_pipeline import ask_question
from app.retrieval.search import (
    semantic_search,
    hybrid_search,
)
from app.retrieval.bm25_search import bm25_search


app = FastAPI(
    title="Enterprise RAG",
    version="1.0.0",
)


# ----------------------------------
# Request Models
# ----------------------------------

class SearchRequest(BaseModel):

    question: str

    top_k: int = Field(
        default=5,
        ge=1,
        le=20
    )

    file_name: str | None = None

    department: str | None = None

    page_label: str | None = None


class ChatRequest(BaseModel):

    question: str


# ----------------------------------
# Root endpoint
# ----------------------------------

@app.get("/")
def root():

    return {
        "message": "Enterprise RAG API is running"
    }


# ----------------------------------
# Health check
# ----------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }


# ----------------------------------
# Load PDFs
# ----------------------------------

@app.get("/load-pdfs")
def load_pdfs():

    documents = load_pdf_documents()

    return {
        "total_documents": len(documents),
        "message": "PDFs loaded successfully",
    }


# ----------------------------------
# Chunk PDFs
# ----------------------------------

@app.get("/chunk-pdfs")
def chunk_pdfs():

    documents = load_pdf_documents()

    nodes = chunk_documents(documents)

    return {
        "total_documents": len(documents),
        "total_chunks": len(nodes),
        "message": "Documents chunked successfully",
    }


# ----------------------------------
# Semantic Search
# ----------------------------------

@app.post("/search")
def search(request: SearchRequest):

    results = semantic_search(
        question=request.question,
        top_k=request.top_k,
        file_name=request.file_name,
        department=request.department,
        page_label=request.page_label,
    )

    return {
        "question": request.question,

        "filters": {
            "file_name": request.file_name,
            "department": request.department,
            "page_label": request.page_label,
        },

        "results": results,
    }


# ----------------------------------
# Chat / RAG
# ----------------------------------

@app.post("/chat")
def chat(request: ChatRequest):

    result = ask_question(
        request.question
    )

    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"],
    }


# ----------------------------------
# Ingest Documents
# ----------------------------------

@app.post("/ingest")
def ingest_documents():

    build_index()

    return {
        "message": "Documents ingested and FAISS index created successfully"
    }

# ----------------------------------
# Hybrid Search
# ----------------------------------
import time

@app.post("/hybrid-search")
def hybrid_search_api(
    request: SearchRequest
):

    print("\n" + "=" * 60)
    print("HYBRID SEARCH API STARTED")
    print("=" * 60)

    start_time = time.time()

    # ----------------------------------
    # 1. Start Hybrid Search
    # ----------------------------------

    print("Calling hybrid_search()...")

    results = hybrid_search(
        question=request.question,
        top_k=request.top_k,
        file_name=request.file_name,
        department=request.department,
        page_label=request.page_label,
    )

    # ----------------------------------
    # 2. Calculate execution time
    # ----------------------------------

    execution_time = time.time() - start_time

    print(
        f"Hybrid search completed in "
        f"{execution_time:.2f} seconds"
    )

    print(
        f"Results returned: {len(results)}"
    )

    print("=" * 60)
    print("HYBRID SEARCH API FINISHED")
    print("=" * 60)

    # ----------------------------------
    # 3. Return response
    # ----------------------------------

    return {

        "question": request.question,

        "search_type": "hybrid",

        "execution_time_seconds": round(
            execution_time,
            2
        ),

        "filters": {

            "file_name": request.file_name,

            "department": request.department,

            "page_label": request.page_label,

        },

        "results": results,

    }
# ----------------------------------
# BM25 Search
# ----------------------------------

@app.post("/bm25-search")
def bm25_search_api(
    request: SearchRequest
):

    results = bm25_search(
        question=request.question,
        top_k=request.top_k,
        file_name=request.file_name,
        department=request.department,
        page_label=request.page_label,
    )

    return {
        "question": request.question,
        "search_type": "BM25",
        "filters": {
            "file_name": request.file_name,
            "department": request.department,
            "page_label": request.page_label,
        },
        "results": results,
    }