from langchain_ollama import ChatOllama


MODEL_NAME = "qwen2.5:3b"
OLLAMA_BASE_URL = "http://localhost:11434"


def get_llm():

    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    return llm