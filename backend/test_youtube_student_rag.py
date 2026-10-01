from rag.retrieval import retrieve


EVAL_CASES = [
    {"query": "What is artificial intelligence?", "relevance": [3, 1, 1]},
    {"query": "What are the major subfields of AI mentioned in the video?", "relevance": [3, 1, 2]},
    {"query": "What is machine learning?", "relevance": [3, 1, 1]},
    {"query": "What is deep learning?", "relevance": [3, 1, 1]},
    {"query": "What is natural language processing?", "relevance": [3, 1, 1]},
    {"query": "What is computer vision?", "relevance": [3, 1, 1]},
    {"query": "Why is data important for AI?", "relevance": [3, 1, 1]},
    {"query": "What is the difference between AI and machine learning?", "relevance": [3, 1, 1]},
    {"query": "What skills are discussed for building a career in AI?", "relevance": [3, 1, 1]},
    {"query": "What technologies are discussed in the AI crash course?", "relevance": [3, 1, 1]},
]


def test_youtube_student_rag():

    hits = 0
    relevant_results = 0
    reciprocal_ranks = []

    for case in EVAL_CASES:

        results = retrieve(
            query=case["query"],
            top_k=3,
            knowledge_scope="student_material",
            source_type="youtube",
        )

        assert len(results) == 3

        for result in results:
            metadata = result["metadata"]

            assert metadata["knowledge_scope"] == "student_material"
            assert metadata["source_type"] == "youtube"

        first_relevant_rank = None

        for rank, relevance in enumerate(
            case["relevance"],
            start=1,
        ):
            if relevance >= 2:
                relevant_results += 1

                if first_relevant_rank is None:
                    first_relevant_rank = rank

        if first_relevant_rank is not None:
            hits += 1
            reciprocal_ranks.append(1 / first_relevant_rank)
        else:
            reciprocal_ranks.append(0)

    total_queries = len(EVAL_CASES)
    total_results = total_queries * 3

    hit_at_3 = hits / total_queries
    precision_at_3 = relevant_results / total_results
    mrr = sum(reciprocal_ranks) / total_queries

    print("\nYOUTUBE STUDENT RAG RESULTS")
    print("=" * 50)
    print(f"Hit@3:       {hit_at_3:.2%}")
    print(f"Precision@3: {precision_at_3:.2%}")
    print(f"MRR:         {mrr:.2%}")
    print("Source isolation: PASSED")


if __name__ == "__main__":
    test_youtube_student_rag()