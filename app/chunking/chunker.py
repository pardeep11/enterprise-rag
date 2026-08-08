from llama_index.core.node_parser import SentenceSplitter


def chunk_documents(documents):

    splitter = SentenceSplitter(
        chunk_size=256,
        chunk_overlap=30,
    )

    nodes = splitter.get_nodes_from_documents(
        documents
    )

    print("=" * 60)
    print("CHUNKING COMPLETE")
    print("=" * 60)

    print(f"Documents : {len(documents)}")
    print(f"Chunks    : {len(nodes)}")

    return nodes