from pathlib import Path
import json

import bm25s


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

BM25_INDEX_DIR = BASE_DIR / "bm25_index"


# ============================================================
# Load BM25 Store
# ============================================================

def load_bm25_store():

    # ----------------------------------
    # Load BM25 index
    # ----------------------------------

    retriever = bm25s.BM25.load(
        str(BM25_INDEX_DIR),
        load_corpus=False
    )

    print('retriever',retriever)
    # ----------------------------------
    # Load document IDs
    # ----------------------------------

    doc_ids = (
        BM25_INDEX_DIR / "doc_ids.txt"
    ).read_text(
        encoding="utf-8"
    ).splitlines()

    print('doc_ids',doc_ids)
    # ----------------------------------
    # Load metadata
    # ----------------------------------

    with open(
        BM25_INDEX_DIR / "doc_metadata.json",
        "r",
        encoding="utf-8"
    ) as file:

        metadata_map = json.load(file)
    print('metadata_map',metadata_map)
    return (
        retriever,
        doc_ids,
        metadata_map
    )


# ============================================================
# BM25 Search
# ============================================================

def bm25_search(
    question: str,
    top_k: int = 5,
    file_name: str | None = None,
    department: str | None = None,
    page_label: str | None = None,
):

    # ----------------------------------
    # Load BM25
    # ----------------------------------

    (
        retriever,
        doc_ids,
        metadata_map
    ) = load_bm25_store()

    # ----------------------------------
    # Tokenize query
    # ----------------------------------

    query_tokens = bm25s.tokenize(
        [question],
        stopwords="en"
    )
    print('Bm25 file')
    print('query_tokensss',query_tokens)
    # ----------------------------------
    # Retrieve candidates
    # ----------------------------------

    candidate_k = min(
        top_k * 4,
        len(doc_ids)
    )

    results, scores = retriever.retrieve(
        query_tokens,
        k=candidate_k
    )

    # ----------------------------------
    # Prepare results
    # ----------------------------------
    print('results',results)
    print('scores',scores)
    final_results = []

    for row_idx, score in zip(
        results[0],
        scores[0]
    ):

        node_id = doc_ids[int(row_idx)]

        meta = metadata_map[node_id]

        # ----------------------------------
        # File filter
        # ----------------------------------

        if file_name:

            if meta["file_name"] != file_name:

                continue

        # ----------------------------------
        # Department filter
        # ----------------------------------

        if department:

            if meta["department"].lower() != department.lower():

                continue

        # ----------------------------------
        # Page filter
        # ----------------------------------

        if page_label:

            if str(meta["page_label"]) != str(page_label):

                continue

        # ----------------------------------
        # Result
        # ----------------------------------

        final_results.append({

            "node_id": node_id,

            "bm25_score": float(score),

            "text": meta["text"],

            "metadata": {

                "file_name": meta["file_name"],

                "file_path": meta["file_path"],

                "page_label": meta["page_label"],

                "department": meta["department"],

            }

        })

        if len(final_results) >= top_k:

            break

    return final_results

if __name__ == "__main__":

    query = "What are the types of leave in Goa?"

    results = bm25_search(
        question=query,
        top_k=3
    )

    print("\n" + "=" * 80)
    print("BM25 SEARCH RESULTS")
    print("=" * 80)

    for result in results:

        print(
            f"\nRank: {result['node_id']}"
        )

        print(
            f"BM25 Score: "
            f"{result['bm25_score']:.4f}"
        )

        print(
            f"File: "
            f"{result['metadata']['file_name']}"
        )

        print(
            f"Page: "
            f"{result['metadata']['page_label']}"
        )

        print(
            f"Department: "
            f"{result['metadata']['department']}"
        )

        print(
            f"Text:\n"
            f"{result['text'][:300]}"
        )

        print("-" * 80)