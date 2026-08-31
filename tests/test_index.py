from fastapi.testclient import TestClient

from faiss_store.index import app

client = TestClient(app)

def test_ingest_documents():
    response = client.post("/ingest")

    print("\nSTATUS:", response.status_code)
    print("\nRESPONSE:", response.json())

    assert response.status_code == 200


if __name__ == "__main__":
    test_ingest_documents()