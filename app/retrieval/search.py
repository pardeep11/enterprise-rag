import faiss
import pickle
import numpy as np

from app.embeddings.embedder import get_embedding


FAISS_INDEX_PATH = "vectorstore/index.faiss"
METADATA_PATH = "vectorstore/metadata.pkl"


def load_vector_store():

    # ----------------------------------
    # Load FAISS index
    # ----------------------------------

    faiss_index = faiss.read_index(
        FAISS_INDEX_PATH
    )

    # ----------------------------------
    # Load metadata
    # ----------------------------------

    with open(
        METADATA_PATH,
        "rb"
    ) as file:

        metadata = pickle.load(file)

    print(
        f"FAISS index loaded: "
        f"{faiss_index.ntotal} vectors"
    )

    print(
        f"Metadata loaded: "
        f"{len(metadata)} chunks"
    )

    return faiss_index, metadata


def semantic_search(
    question: str,
    top_k: int = 5
):

    # ----------------------------------
    # 1. Load vector store
    # ----------------------------------

    faiss_index, metadata = load_vector_store()

    # ----------------------------------
    # 2. Embed question
    # ----------------------------------

    query_embedding = get_embedding(
        question
    )

    # ----------------------------------
    # 3. Convert to NumPy
    # ----------------------------------

    query_vector = np.array(
        [query_embedding],
        dtype=np.float32
    )

    # ----------------------------------
    # 4. FAISS similarity search
    # ----------------------------------

    distances, indices = faiss_index.search(
        query_vector,
        top_k
    )

    # ----------------------------------
    # 5. Get matching chunks
    # ----------------------------------

    results = []

    for rank, (distance, index_id) in enumerate(
        zip(distances[0], indices[0]),
        start=1
    ):

        if index_id == -1:
            continue

        result = {
            "rank": rank,
            "score": float(distance),
            "text": metadata[index_id]["text"],
            "metadata": metadata[index_id]["metadata"],
        }

        results.append(result)

    return results