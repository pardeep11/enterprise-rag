import os

from llama_index.core import SimpleDirectoryReader


FILES = [
    "data/pdf/hr/Leave_Policy.pdf",
    "data/pdf/cloud/aws_policy.pdf",
]


def load_pdf_documents():
    """
    Load and return documents from PDF files.
    """

    documents = []

    for file_path in FILES:

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        try:

            docs = SimpleDirectoryReader(
                input_files=[file_path]
            ).load_data()

            documents.extend(docs)

            print(f"Loaded: {file_path}")
            print(f"   Pages: {len(docs)}")

        except Exception as e:

            print(f"Failed to load: {file_path}")
            print(f"   Error: {e}")

    return documents