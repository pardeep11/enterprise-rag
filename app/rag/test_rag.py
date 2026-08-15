from app.rag.rag_pipeline import ask_question


def main():

    question = input(
        "Enter your question: "
    )

    result = ask_question(
        question
    )

    print()
    print("=" * 60)
    print("QUESTION")
    print("=" * 60)

    print(
        result["question"]
    )

    print()
    print("=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(
        result["answer"]
    )

    print()
    print("=" * 60)
    print("SOURCES")
    print("=" * 60)

    for source in result["sources"]:

        print()
        print(
            f"Rank: {source['rank']}"
        )

        print(
            f"Score: {source['score']}"
        )

        print(
            f"Metadata: {source['metadata']}"
        )


if __name__ == "__main__":
    main()