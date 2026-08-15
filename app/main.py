from fastapi import FastAPI
from pydantic import BaseModel

from app.loaders.pdf_loader import load_pdf_documents
from app.chunking.chunker import chunk_documents
from app.vectorstore.index import build_index
from app.retrieval.search import semantic_search


app = FastAPI(
    title="Enterprise RAG",
    version="1.0.0",
)


# ----------------------------------
# Request Models
# ----------------------------------

class SearchRequest(BaseModel):

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
# Build FAISS index
# ----------------------------------

@app.get("/build-index")
def create_index():

    build_index()

    return {
        "message": "Embeddings generated and vector index created successfully",
    }


# ----------------------------------
# Semantic Search
# ----------------------------------

@app.post("/search")
def search(request: SearchRequest):

    results = semantic_search(
        request.question
    )

    return {
        "question": request.question,
        "results": results,
    }