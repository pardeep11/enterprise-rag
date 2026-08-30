from sentence_transformers import CrossEncoder

# Load model globally on app startup
# This downloads/loads the ~90MB weights into memory once
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_results(query: str, documents: list[dict], top_k: int) -> list[dict]:
    """
    Reranks candidate documents based on semantic relevance to the query.
    
    Expected doc format: [{'text': '...', 'file_name': '...', ...}]
    """
    if not documents:
        return []

    # 1. Prepare (query, text) pairs for the model
    # Note: If your candidate payload uses a key other than 'text' (e.g. 'content'), update it here.
    pairs = [[query, doc.get("text", "")] for doc in documents]

    # 2. Get relevance scores (logits/scores from MS-MARCO)
    scores = reranker_model.predict(pairs)

    # 3. Attach scores back to the original document objects
    for doc, score in zip(documents, scores):
        doc["rerank_score"] = float(score)

    # 4. Sort documents descending by score and return top_k
    sorted_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
    return sorted_docs[:top_k]