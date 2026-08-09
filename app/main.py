from fastapi import FastAPI

from app.loaders.pdf_loader import load_pdf_documents
from app.chunking.chunker import chunk_documents
from app.vectorstore.index import build_index


app = FastAPI(
    title="Enterprise RAG",
    version="1.0.0",
)


@app.get("/")
def root():

    return {
        "message": "Enterprise RAG API is running"
    }


@app.get("/load-pdfs")
def load_pdfs():

    documents = load_pdf_documents()

    return {
        "total_documents": len(documents),
        "message": "PDFs loaded successfully",
    }


@app.get("/chunk-pdfs")
def chunk_pdfs():

    documents = load_pdf_documents()

    nodes = chunk_documents(documents)

    return {
        "total_documents": len(documents),
        "total_chunks": len(nodes),
        "message": "Documents chunked successfully",
    }


@app.get("/build-index")
def create_index():

    build_index()

    return {
        "message": "FAISS index created successfully",
    }