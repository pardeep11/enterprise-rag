import time
from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.loaders.pdf_loader import load_pdf_documents
from app.chunking.chunker import chunk_documents
from app.faiss_store.index import build_index
from app.retrieval.reranking import rerank_results
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


class HybridSearchRequest(BaseModel):
    question: str

    top_k: int = Field(
        default=5,
        ge=1,
        le=20
    )

    file_name: str | None = None
    department: str | None = None
    page_label: str | None = None

    enable_rerank: bool = True

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
@app.post("/hybrid-search")
def hybrid_search_api(
    request: SearchRequest
):

    print("\n" + "=" * 60)
    print("HYBRID SEARCH API STARTED")
    print("=" * 60)

    start_time = time.time()

    # ----------------------------------
    # 1. Hybrid Search
    # ----------------------------------

    print(
        f"Question: {request.question}"
    )

    print(
        f"Retrieving top {request.top_k} results..."
    )

    results = hybrid_search(
        question=request.question,
        top_k=request.top_k,
        file_name=request.file_name,
        department=request.department,
        page_label=request.page_label,
    )

    # ----------------------------------
    # 2. Execution Time
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
    # 3. Return Response
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

        "result_count": len(results),

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


@app.post("/hybrid-rerank")
def hybrid_rerank_api(
    request: HybridSearchRequest
):

    print("\n" + "=" * 60)
    print("HYBRID SEARCH + RERANK API STARTED")
    print("=" * 60)
    print('request:', request)
    start_time = time.time()

    # ----------------------------------
    # 1. Retrieve candidates
    # ----------------------------------

    candidate_k = (
        request.top_k * 3
        if request.enable_rerank
        else request.top_k
    )

    print(
        f"Calling hybrid_search() "
        f"with top {candidate_k} candidates..."
    )

    raw_results = hybrid_search(
        question=request.question,
        top_k=candidate_k,
        file_name=request.file_name,
        department=request.department,
        page_label=request.page_label,
    )

    retrieval_time = time.time() - start_time

    print(
        f"Hybrid search retrieved "
        f"{len(raw_results)} candidates "
        f"in {retrieval_time:.2f}s"
    )

    print('raw_results',raw_results[0])

    # ----------------------------------
    # 2. Reranking
    # ----------------------------------

    rerank_start = time.time()

    if request.enable_rerank and raw_results:

        print(
            "Running CrossEncoder reranker..."
        )

        final_results = rerank_results(
            query=request.question,
            documents=raw_results,
            top_k=request.top_k,
        )

        # Reassign final ranking
        for rank, result in enumerate(
            final_results,
            start=1
        ):
            result["rank"] = rank

    else:

        final_results = raw_results[
            :request.top_k
        ]

    rerank_time = time.time() - rerank_start

    # ----------------------------------
    # 3. Total execution time
    # ----------------------------------

    total_execution_time = (
        time.time() - start_time
    )

    print(
        f"Reranking completed in "
        f"{rerank_time:.2f}s"
    )

    print(
        f"Final results returned: "
        f"{len(final_results)}"
    )

    print("=" * 60)
    print("HYBRID SEARCH + RERANK API FINISHED")
    print("=" * 60)

    # ----------------------------------
    # 4. Response
    # ----------------------------------

    return {

        "question": request.question,

        "search_type": (
            "hybrid_with_rerank"
            if request.enable_rerank
            else "hybrid"
        ),

        "candidate_count": len(raw_results),

        "result_count": len(final_results),

        "execution_time_seconds": round(
            total_execution_time,
            2
        ),

        "timing_breakdown": {

            "retrieval_seconds": round(
                retrieval_time,
                2
            ),

            "rerank_seconds": round(
                rerank_time,
                2
            ),

        },

        "filters": {

            "file_name": request.file_name,

            "department": request.department,

            "page_label": request.page_label,

        },

        "results": final_results,
    }