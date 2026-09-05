import json
from typing import Dict, List
from app.retrieval.search import (
    semantic_search,
    hybrid_search,
    mmr_search,          
)

from app.retrieval.reranking import rerank_results

# Load Ground Truth
with open("data/evaluation/auto_ground_truth.json", "r") as f:
    ground_truth = json.load(f)["questions"]
    print(f"Number of ground truth questions: {len(ground_truth)}")
    print(ground_truth[0].get('question'), ' ground_truth')
    print(ground_truth[0].get('source_metadata'), ' ground_truth')
    sc=ground_truth[0].get('source_metadata')
    print(sc.get('file_name'), ' ground_truth')


def evaluate_retrieval(
    retriever_func,
    top_k: int = 3,
    method_name: str = ""
):

    print("GROUND TRUTH TYPE:", type(ground_truth))
    print("TOTAL QUESTIONS:", len(ground_truth))
    print("QUESTION IDs:", [item["question_id"] for item in ground_truth])

    total_queries = len(ground_truth)
    hits = 0
    mrr_score = 0.0
    failed_queries = []

    for item in ground_truth:

        query_question = item["question"]

        expected_ids = set(item["ground_truth_chunk_ids"])

        retrieved_chunks = retriever_func(
            question=query_question,
            top_k=top_k,
            file_name=None,
            department=None,
            page_label=None,
        )

        retrieved_ids = [
            chunk.get("node_id")
            for chunk in retrieved_chunks
        ]

        print("\nQuestion:", query_question)
        print("Expected IDs:", expected_ids)
        print("Retrieved IDs:", retrieved_ids)

        # Hit@K
        if any(cid in retrieved_ids for cid in expected_ids):
            hits += 1

        # MRR
        found = False

        for rank, cid in enumerate(retrieved_ids, start=1):
            if cid in expected_ids:
                mrr_score += 1.0 / rank
                print(f"Correct chunk found at Rank {rank}")
                found = True
                break

        if not found:
            print("Correct chunk NOT found")
            failed_queries.append({
                    "question_id": item["question_id"],
                    "question": query_question,
                    "expected_ids": list(expected_ids),
                    "retrieved_ids": retrieved_ids
                })

    hit_rate = hits / total_queries
    mrr = mrr_score / total_queries

    print(f"\n--- {method_name} Results (Top {top_k}) ---")
    print(f"Hit Rate: {hit_rate * 100:.2f}%")
    print(f"MRR:      {mrr:.4f}")

    print(f"\n=== FAILED QUERIES: {method_name} ===")

    for failure in failed_queries:
        print(f"\nQuestion ID: {failure['question_id']}")
        print(f"Question: {failure['question']}")
        print(f"Expected IDs: {failure['expected_ids']}")
        print(f"Retrieved IDs: {failure['retrieved_ids']}")

    return {
        "total_queries": total_queries,
        "top_k": top_k,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "failed_queries": failed_queries
    }

def hybrid_rerank_search(
    question,
    top_k=3,
    file_name=None,
    department=None,
    page_label=None,
):
    # 1. Retrieve more candidates
    candidate_k = top_k * 3

    raw_results = hybrid_search(
        question=question,
        top_k=candidate_k,
        file_name=file_name,
        department=department,
        page_label=page_label,
    )

    # DEBUG
    if question in [
        "How many sections does the table of contents have in the AWS Cloud Adoption Framework Whitepaper?",
        "What does the AWS CAF identify and provide prescriptive guidance for?"
    ]:
        print("\nDEBUG:", question)
        print("HYBRID CANDIDATE IDs:")
        for result in raw_results:
            print(result.get("node_id"))

    # 2. Rerank candidates
    final_results = rerank_results(
        query=question,
        documents=raw_results,
        top_k=top_k,
    )

    # 3. Return final results
    return final_results


def generate_report(
    hybrid_result,
    rerank_result,
    mmr_result
):

    report = {
        "dataset_size": hybrid_result["total_queries"],
        "top_k": hybrid_result["top_k"],

        "results": {

            "hybrid_search": {
                "hit_at_3": hybrid_result["hit_rate"],
                "mrr": hybrid_result["mrr"]
            },

            "hybrid_reranking": {
                "hit_at_3": rerank_result["hit_rate"],
                "mrr": rerank_result["mrr"]
            },

            "mmr_search": {
                "hit_at_3": mmr_result["hit_rate"],
                "mrr": mmr_result["mrr"]
            }
        },

        "improvement": {
            "reranking_mrr_absolute":
                rerank_result["mrr"]
                - hybrid_result["mrr"],

            "mmr_mrr_absolute":
                mmr_result["mrr"]
                - hybrid_result["mrr"]
        }
    }

    with open(
        "evaluation_report.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    print("\nEvaluation report generated:")
    print("evaluation_report.json")


if __name__ == "__main__":

    print("\n===== HYBRID SEARCH =====")

    hybrid_result = evaluate_retrieval(
        hybrid_search,
        top_k=3,
        method_name="Hybrid Search"
    )

    print("\n===== HYBRID + RERANKING =====")

    rerank_result = evaluate_retrieval(
        hybrid_rerank_search,
        top_k=3,
        method_name="Hybrid + Reranking"
    )

    print("\n===== MMR SEARCH =====")

    mmr_result = evaluate_retrieval(
        mmr_search,
        top_k=3,
        method_name="MMR"
    )

    print("\nHYBRID RESULT:")
    print(hybrid_result)

    print("\nRERANK RESULT:")
    print(rerank_result)

    print("\nMMR RESULT:")
    print(mmr_result)

    generate_report(
        hybrid_result,
        rerank_result,
        mmr_result
    )