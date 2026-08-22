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


# ==================================
# Metadata Filter
# ==================================

def matches_filter(
    metadata,
    file_name=None,
    department=None,
    page_label=None,
):

    # ----------------------------------
    # Filter by file name
    # ----------------------------------

    if file_name:

        if metadata.get("file_name") != file_name:
            return False

    # ----------------------------------
    # Filter by department
    # ----------------------------------

    if department:

        file_path = metadata.get(
            "file_path",
            ""
        ).lower()

        department_path = (
            f"/{department.lower()}/"
        )

        if department_path not in file_path:
            return False

    # ----------------------------------
    # Filter by page
    # ----------------------------------

    if page_label:

        if str(
            metadata.get("page_label")
        ) != str(page_label):

            return False

    return True


# ==================================
# Semantic Search
# ==================================

def semantic_search(
    question: str,
    top_k: int = 5,
    file_name: str | None = None,
    department: str | None = None,
    page_label: str | None = None,
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
    # 4. Search more candidates
    # ----------------------------------

    candidate_k = min(
        top_k * 4,
        faiss_index.ntotal
    )

    distances, indices = faiss_index.search(
        query_vector,
        candidate_k
    )

    # ----------------------------------
    # 5. Metadata filtering
    # ----------------------------------

    results = []

    for distance, index_id in zip(
        distances[0],
        indices[0]
    ):

        if index_id == -1:
            continue

        item_metadata = metadata[index_id]["metadata"]

        # ----------------------------------
        # Apply metadata filter
        # ----------------------------------

        if not matches_filter(
            item_metadata,
            file_name=file_name,
            department=department,
            page_label=page_label,
        ):
            continue

        # ----------------------------------
        # Add result
        # ----------------------------------

        result = {
            "rank": len(results) + 1,
            "score": float(distance),
            "text": metadata[index_id]["text"],
            "metadata": item_metadata,
        }

        results.append(result)

        # ----------------------------------
        # Stop after top_k results
        # ----------------------------------

        if len(results) >= top_k:
            break

    return results