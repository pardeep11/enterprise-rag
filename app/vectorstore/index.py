import numpy as np

from app.loaders.pdf_loader import load_pdf_documents
from app.chunking.chunker import chunk_documents
from app.embeddings.embedder import get_embedding


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

    return nodes, embedding_matrix