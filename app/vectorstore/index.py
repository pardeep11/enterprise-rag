import faiss
import pickle
import numpy as np
from pathlib import Path

from app.loaders.pdf_loader import load_pdf_documents
from app.chunking.chunker import chunk_documents
from app.embeddings.embedder import get_embedding


# ----------------------------------
# FAISS storage paths
# ----------------------------------

VECTORSTORE_DIR = Path("vectorstore")

FAISS_INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
METADATA_PATH = VECTORSTORE_DIR / "metadata.pkl"


def build_index():

    # ----------------------------------
    # 1. Load documents
    # ----------------------------------

    documents = load_pdf_documents()

    print(f"Documents loaded: {len(documents)}")

    # ----------------------------------
    # 2. Create chunks
    # ----------------------------------

    nodes = chunk_documents(documents)

    print(f"Chunks created: {len(nodes)}")

    # ----------------------------------
    # 3. Generate embeddings
    # ----------------------------------

    embeddings = []

    print()
    print("Generating embeddings...")

    for i, node in enumerate(nodes, start=1):

        embedding = get_embedding(node.text)

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

    print(f"Documents : {len(documents)}")
    print(f"Chunks    : {len(nodes)}")
    print(f"Vectors   : {len(embeddings)}")
    print(f"Dimensions: {embedding_matrix.shape[1]}")
    print(f"Matrix    : {embedding_matrix.shape}")

    # ----------------------------------
    # 5. Create FAISS CPU index
    # ----------------------------------

    dimension = embedding_matrix.shape[1]

    print()
    print("Creating FAISS CPU index...")

    faiss_index = faiss.IndexFlatL2(dimension)

    # ----------------------------------
    # 6. Add vectors to FAISS
    # ----------------------------------

    faiss_index.add(embedding_matrix)

    print(
        f"Vectors stored in FAISS: "
        f"{faiss_index.ntotal}"
    )

    # ----------------------------------
    # 7. Create vector_store directory
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
                "text": node.text,
                "metadata": node.metadata,
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
    print("MILESTONE 5 COMPLETE")
    print("=" * 60)

    print(f"FAISS vectors : {faiss_index.ntotal}")
    print(f"Dimensions    : {dimension}")
    print(f"Index file    : {FAISS_INDEX_PATH}")
    print(f"Metadata file : {METADATA_PATH}")

    return faiss_index, metadata