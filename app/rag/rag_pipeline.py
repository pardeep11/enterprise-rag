from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.llm.ollama_llm import get_llm
from app.retrieval.search import semantic_search


# ----------------------------------
# Prompt
# ----------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful enterprise knowledge assistant.

Answer the user's question using ONLY the
provided context.

If the answer is not available in the context,
say:

"I don't have enough information in the
provided documents."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""
)


# ----------------------------------
# LLM
# ----------------------------------

llm = get_llm()


# ----------------------------------
# LCEL Pipeline
# ----------------------------------

rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)


def format_context(results):

    context_parts = []

    for result in results:

        context_parts.append(
            result["text"]
        )

    return "\n\n".join(
        context_parts
    )


def ask_question(question: str):

    # ----------------------------------
    # 1. Semantic search
    # ----------------------------------

    results = semantic_search(
        question,
        top_k=5
    )

    # ----------------------------------
    # 2. Build context
    # ----------------------------------

    context = format_context(
        results
    )

    # ----------------------------------
    # 3. Send context + question to LLM
    # ----------------------------------

    answer = rag_chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return {
        "question": question,
        "answer": answer,
        "sources": results,
    }