from app.retrieval.search import semantic_search


def main():

    question = input(
        "Enter your question: "
    )

    results = semantic_search(
        question,
        top_k=5
    )

    print()
    print("=" * 60)
    print("SEMANTIC SEARCH RESULTS")
    print("=" * 60)

    for result in results:

        print()
        print(
            f"Rank : {result['rank']}"
        )

        print(
            f"Score: {result['score']}"
        )

        print(
            f"Text : {result['text']}"
        )

        print(
            f"Metadata: {result['metadata']}"
        )


if __name__ == "__main__":
    main()