import ollama


MODEL_NAME = "nomic-embed-text"
OLLAMA_HOST = "http://localhost:11434"


def get_embedding(text: str) -> list[float]:

    client = ollama.Client(
        host=OLLAMA_HOST
    )

    response = client.embed(
        model=MODEL_NAME,
        input=text,
    )

    return response["embeddings"][0]