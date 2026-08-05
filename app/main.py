from fastapi import FastAPI

app = FastAPI(
    title="Enterprise RAG",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "Enterprise RAG API is running"}
