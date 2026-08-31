from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_hybrid_rerank_search():

    payload = {
        "question": "What are the types of leave in Goa?",
        "top_k": 5,
        "enable_rerank": True
    }

    print("\nQUESTION:", payload["question"])
    print("TOP_K:", payload["top_k"])
    print("ENABLE_RERANK:", payload["enable_rerank"])

    response = client.post(
        "/hybrid-rerank",
        json=payload
    )

    print("\nSTATUS CODE:")
    print(response.status_code)

    print("\nRESPONSE:")
    print(response.json())

    assert response.status_code == 200

    data = response.json()

    assert data["search_type"] == "hybrid_with_rerank"
    assert len(data["results"]) <= payload["top_k"]


if __name__ == "__main__":
    test_hybrid_rerank_search()