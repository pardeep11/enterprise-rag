import faiss
import pickle
import numpy as np

from app.embeddings.embedder import get_embedding
from app.retrieval.bm25_search import bm25_search


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

    print('----------semantic search function data-------')
    print('candidate_k ',candidate_k)
    print('query_vector ',query_vector)
    print('distances ',distances)
    print('indices ',indices)
    print('faiss_index.ntotal ',faiss_index.ntotal)
    print('top_k ',top_k)
    print('metadata ',metadata)


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
                    "node_id": metadata[index_id]["node_id"],
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

# ==================================
# Hybrid Search
# ==================================

def hybrid_search(
    question: str,
    top_k: int = 5,
    file_name: str | None = None,
    department: str | None = None,
    page_label: str | None = None,
):

    # ----------------------------------
    # Retrieve more candidates
    # ----------------------------------

    candidate_k = top_k * 4

    # ----------------------------------
    # 1. Semantic Search
    # ----------------------------------

    semantic_results = semantic_search(
        question=question,
        top_k=candidate_k,
        file_name=file_name,
        department=department,
        page_label=page_label,
    )

    print('semantic_results',semantic_results)
    # ----------------------------------
    # 2. BM25 Search
    # ----------------------------------

    keyword_results = bm25_search(
        question=question,
        top_k=candidate_k,
        file_name=file_name,
        department=department,
        page_label=page_label,
    )

    # ----------------------------------
    # 3. Reciprocal Rank Fusion
    # ----------------------------------
    print('keyword_results:',keyword_results)
    fused_scores = {}

    result_map = {}

    RRF_K = 60

    # ----------------------------------
    # FAISS ranking
    # ----------------------------------

    for rank, result in enumerate(
        semantic_results,
        start=1
    ):

        node_id = result["node_id"]

        fused_scores[node_id] = (
            fused_scores.get(node_id, 0)
            + 1 / (RRF_K + rank)
        )

        result_map[node_id] = result

    # ----------------------------------
    # BM25 ranking
    # ----------------------------------
    print('result_map',result_map)
    print('fused_scores::',fused_scores)
    for rank, result in enumerate(
        keyword_results,
        start=1
    ):

        node_id = result["node_id"]

        fused_scores[node_id] = (
            fused_scores.get(node_id, 0)
            + 1 / (RRF_K + rank)
        )

        # Keep existing FAISS result if
        # BM25 found the same chunk
        if node_id not in result_map:

            result_map[node_id] = result

    # ----------------------------------
    # Sort by RRF score
    # ----------------------------------

    ranked_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True
    )

    print('ranked_ids',ranked_ids)
    # ----------------------------------
    # Final results
    # ----------------------------------

    final_results = []

    for rank, node_id in enumerate(
        ranked_ids[:top_k],
        start=1
    ):

        result = result_map[node_id]

        final_results.append({

            "rank": rank,

            "node_id": node_id,

            "hybrid_score": round(
                fused_scores[node_id],
                6
            ),

            "text": result["text"],

            "metadata": result["metadata"],

        })

    return final_results

# ==================================
# MMR Search
# ==================================

def mmr_search(
    question: str,
    top_k: int = 5,
    candidate_k: int | None = None,
    lambda_param: float = 0.7,
    file_name: str | None = None,
    department: str | None = None,
    page_label: str | None = None,
):
    """
    Retrieve documents using Maximal Marginal Relevance (MMR).

    MMR balances:
        1. Relevance to the query
        2. Diversity between selected chunks

    lambda_param:
        1.0 -> only relevance
        0.0 -> only diversity
        0.7 -> good starting point
    """

    # ----------------------------------
    # 1. Load vector store
    # ----------------------------------

    faiss_index, metadata = load_vector_store()

    # ----------------------------------
    # 2. Embed question
    # ----------------------------------

    query_embedding = get_embedding(question)

    query_vector = np.array(
        query_embedding,
        dtype=np.float32
    )

    # Check query vector
    query_norm = np.linalg.norm(query_vector)

    if query_norm == 0:
        return []

    # ----------------------------------
    # 3. Candidate count
    # ----------------------------------

    if candidate_k is None:
        candidate_k = min(
            top_k * 4,
            faiss_index.ntotal
        )
    else:
        candidate_k = min(
            candidate_k,
            faiss_index.ntotal
        )

    # ----------------------------------
    # 4. FAISS candidate search
    # ----------------------------------

    # Use ORIGINAL query vector for FAISS
    query_for_faiss = np.array(
        [query_vector],
        dtype=np.float32
    )

    distances, indices = faiss_index.search(
        query_for_faiss,
        candidate_k
    )

    # Normalize ONLY for MMR calculation
    query_vector = query_vector / query_norm

    # ----------------------------------
    # 5. Collect valid candidates
    # ----------------------------------

    candidates = []

    for distance, index_id in zip(
        distances[0],
        indices[0]
    ):

        if index_id == -1:
            continue

        item_metadata = metadata[index_id]["metadata"]

        # Apply metadata filtering
        if not matches_filter(
            item_metadata,
            file_name=file_name,
            department=department,
            page_label=page_label,
        ):
            continue

        candidates.append(
            {
                "index_id": index_id,
                "distance": float(distance),
                "text": metadata[index_id]["text"],
                "node_id": metadata[index_id]["node_id"],
                "metadata": item_metadata,
            }
        )

    if not candidates:
        return []

    # ----------------------------------
    # 6. Get candidate vectors
    # ----------------------------------

    candidate_vectors = []

    for candidate in candidates:

        # Convert FAISS/NumPy index to Python int
        vector = faiss_index.reconstruct(
            int(candidate["index_id"])
        )

        vector = np.array(
            vector,
            dtype=np.float32
        )

        # Normalize candidate vector
        norm = np.linalg.norm(vector)

        if norm != 0:
            vector = vector / norm

        candidate_vectors.append(vector)

    candidate_vectors = np.array(
        candidate_vectors,
        dtype=np.float32
    )

    # ----------------------------------
    # 7. Calculate query relevance
    # ----------------------------------

    query_relevance = np.dot(
        candidate_vectors,
        query_vector
    )

    # ----------------------------------
    # 8. MMR selection
    # ----------------------------------

    selected_indices = []

    remaining_indices = list(
        range(len(candidates))
    )

    while (
        remaining_indices
        and len(selected_indices) < top_k
    ):

        best_candidate = None
        best_score = -float("inf")

        for candidate_idx in remaining_indices:

            relevance = query_relevance[
                candidate_idx
            ]

            # First document
            if not selected_indices:

                diversity_penalty = 0.0

            else:

                selected_vectors = (
                    candidate_vectors[
                        selected_indices
                    ]
                )

                candidate_vector = (
                    candidate_vectors[
                        candidate_idx
                    ]
                )

                similarities = np.dot(
                    selected_vectors,
                    candidate_vector
                )

                diversity_penalty = np.max(
                    similarities
                )

            # ----------------------------------
            # MMR formula
            # ----------------------------------

            mmr_score = (
                lambda_param * relevance
                -
                (1 - lambda_param)
                * diversity_penalty
            )

            if mmr_score > best_score:

                best_score = mmr_score
                best_candidate = candidate_idx

        # Add best candidate
        selected_indices.append(
            best_candidate
        )

        remaining_indices.remove(
            best_candidate
        )

    # ----------------------------------
    # 9. Build final results
    # ----------------------------------

    results = []

    for rank, candidate_idx in enumerate(
        selected_indices,
        start=1
    ):

        candidate = candidates[
            candidate_idx
        ]

        results.append(
            {
                "rank": rank,
                "node_id": candidate["node_id"],
                "score": float(
                    query_relevance[
                        candidate_idx
                    ]
                ),
                "text": candidate["text"],
                "metadata": candidate["metadata"],
            }
        )

    return results