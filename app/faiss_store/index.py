import faiss
import pickle
import numpy as np
from pathlib import Path
import uuid
from typing import List, Dict, Any

from app.loaders.pdf_loader import load_pdf_documents
from app.chunking.chunker import chunk_documents
from app.embeddings.embedder import get_embedding


# ----------------------------------
# FAISS storage paths
# ----------------------------------

VECTORSTORE_DIR = Path("vectorstore")

FAISS_INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
METADATA_PATH = VECTORSTORE_DIR / "metadata.pkl"


# ----------------------------------
# Process chunks
# ----------------------------------

def process_chunks(
    raw_chunks: List[Any],
) -> List[Dict[str, Any]]:
    """
    Normalize LlamaIndex TextNode objects or dictionaries
    into a consistent chunk dictionary format.
    """

    processed_chunks = []

    for node in raw_chunks:

        # ----------------------------------
        # Dictionary-based chunk
        # ----------------------------------

        if isinstance(node, dict):

            chunk_id = (
                node.get("chunk_id")
                or node.get("id_")
                or node.get("id")
                or str(uuid.uuid4())
            )

            text_content = node.get("text", "")

            metadata = node.get(
                "metadata",
                {}
            )

        # ----------------------------------
        # LlamaIndex TextNode
        # ----------------------------------

        else:

            node_metadata = getattr(
                node,
                "metadata",
                {}
            )

            chunk_id = (
                getattr(node, "id_", None)
                or (
                    node_metadata.get("id")
                    if isinstance(node_metadata, dict)
                    else None
                )
                or str(uuid.uuid4())
            )

            text_content = getattr(
                node,
                "text",
                ""
            )

            metadata = node_metadata

        # ----------------------------------
        # Skip empty chunks
        # ----------------------------------

        if not text_content or not text_content.strip():
            continue

        processed_chunks.append(
            {
                "chunk_id": chunk_id,
                "text": text_content,
                "metadata": metadata,
            }
        )

    return processed_chunks


# ----------------------------------
# Build FAISS index
# ----------------------------------

def build_index():

    # ----------------------------------
    # 1. Load documents
    # ----------------------------------

    documents = load_pdf_documents()

    print(
        f"Documents loaded: {len(documents)}"
    )

    # ----------------------------------
    # 2. Create chunks
    # ----------------------------------

    raw_chunks = chunk_documents(documents)

    print(
        f"Raw chunks created: {len(raw_chunks)}"
    )

    # Normalize chunks
    nodes = process_chunks(raw_chunks)


    print(
        f"Processed chunks: {len(nodes)}"
    )

    # ----------------------------------
    # 3. Generate embeddings
    # ----------------------------------

    embeddings = []

    print()
    print("Generating embeddings...")

    for i, node in enumerate(
        nodes,
        start=1
    ):

        text = node["text"]

        embedding = get_embedding(text)

        embeddings.append(embedding)

        print(
            f"Embedding {i}/{len(nodes)} "
            f"-> {len(embedding)} dimensions"
        )

    # ----------------------------------
    # 4. Convert embeddings to NumPy
    # ----------------------------------

    embedding_matrix = np.array(
        embeddings,
        dtype=np.float32,
    )

    print()
    print("================================")
    print("EMBEDDING COMPLETE")
    print("================================")

    print(
        f"Documents : {len(documents)}"
    )

    print(
        f"Chunks    : {len(nodes)}"
    )

    print(
        f"Vectors   : {len(embeddings)}"
    )

    print(
        f"Dimensions: {embedding_matrix.shape[1]}"
    )

    print(
        f"Matrix    : {embedding_matrix.shape}"
    )

    # ----------------------------------
    # 5. Create FAISS CPU index
    # ----------------------------------

    dimension = embedding_matrix.shape[1]

    print()
    print("Creating FAISS CPU index...")

    faiss_index = faiss.IndexFlatL2(
        dimension
    )

    # ----------------------------------
    # 6. Add vectors to FAISS
    # ----------------------------------

    faiss_index.add(
        embedding_matrix
    )

    print(
        f"Vectors stored in FAISS: "
        f"{faiss_index.ntotal}"
    )

    # ----------------------------------
    # 7. Create vectorstore directory
    # ----------------------------------

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------------
    # 8. Save FAISS index
    # ----------------------------------

    faiss.write_index(
        faiss_index,
        str(FAISS_INDEX_PATH)
    )

    print(
        f"FAISS index saved: "
        f"{FAISS_INDEX_PATH}"
    )

    # ----------------------------------
    # 9. Save chunk metadata
    # ----------------------------------

    metadata = []

    for node in nodes:

        metadata.append(
            {
                "text": node["text"],
                "node_id": node["chunk_id"],
                "metadata": node["metadata"],
            }
        )

    with open(
        METADATA_PATH,
        "wb"
    ) as file:

        pickle.dump(
            metadata,
            file
        )

    print(
        f"Metadata saved: "
        f"{METADATA_PATH}"
    )

    # ----------------------------------
    # 10. Final output
    # ----------------------------------

    print()
    print("=" * 60)
    print("FAISS INDEX BUILD COMPLETE")
    print("=" * 60)

    print(
        f"FAISS vectors : {faiss_index.ntotal}"
    )

    print(
        f"Dimensions    : {dimension}"
    )

    print(
        f"Index file    : {FAISS_INDEX_PATH}"
    )

    print(
        f"Metadata file : {METADATA_PATH}"
    )

    return faiss_index, metadata


import json
import requests
from pathlib import Path


def generate_ground_truth_questions(
    nodes,
    output_file="data/evaluation/auto_ground_truth.json",
    total_questions=20,
):
    """
    TEMPORARY:
    Generate Ground Truth questions automatically from indexed chunks.

    Each generated question is linked to the exact chunk used
    to generate the question.
    """

    output_path = Path(output_file)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Select chunks
    selected_nodes = nodes[:total_questions]

    questions = []

    for index, node in enumerate(
        selected_nodes,
        start=1
    ):
        chunk_id = node["chunk_id"]
        text = node["text"]
        metadata = node["metadata"]

        print(
            f"\nGenerating question "
            f"{index}/{len(selected_nodes)}"
        )

        prompt = f"""
You are creating an evaluation dataset for an Enterprise RAG system.

Read the following document chunk and create ONE question
that can be answered directly from this chunk.

Rules:
1. The question must be specific to this chunk.
2. The answer must be supported ONLY by this chunk.
3. Do not use information outside the chunk.
4. Do not mention the chunk, document, or source.
5. Return ONLY valid JSON.
6. The answer should be concise but complete.

Document chunk:
----------------
{text}
----------------

Return exactly:

{{
    "question": "...",
    "ground_truth_answer": "..."
}}
"""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:7b",
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=120,
        )

        response.raise_for_status()

        result = response.json()

        generated = json.loads(
            result["response"]
        )

        question_item = {
            "question_id": f"Q{index:03d}",
            "question": generated["question"],
            "ground_truth_answer": generated[
                "ground_truth_answer"
            ],
            "ground_truth_chunk_ids": [
                chunk_id
            ],
            "source_metadata": {
                "file_name": metadata.get(
                    "file_name"
                ),
                "file_path": metadata.get(
                    "file_path"
                ),
                "page_label": str(
                    metadata.get("page_label", "")
                ),
                "file_type": metadata.get(
                    "file_type",
                    "application/pdf"
                ),
            },
        }

        questions.append(question_item)

        print(
            "Question:",
            generated["question"]
        )

    ground_truth = {
        "questions": questions
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            ground_truth,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nGround Truth generated:"
        f"\n{output_path}"
    )

    print(
        f"Total questions: {len(questions)}"
    )

    return ground_truth