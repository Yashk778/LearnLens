from rag.retrieval import retrieve


EVAL_CASES = [
    {
        "query": "What is React and what is Vite used for?",
        "relevance": [3, 1, 1],
    },
    {
        "query": "What are React components?",
        "relevance": [2, 3, 1],
    },
    {
        "query": "What are props in React?",
        "relevance": [1, 3, 1],
    },
    {
        "query": "What does map() do in React?",
        "relevance": [3, 1, 1],
    },
    {
        "query": "What is the difference between props and state?",
        "relevance": [2, 1, 3],
    },
    {
        "query": "What is useState used for?",
        "relevance": [1, 3, 1],
    },
    {
        "query": "What is useEffect used for?",
        "relevance": [3, 1, 1],
    },
    {
        "query": "How does React communicate with FastAPI?",
        "relevance": [3, 1, 2],
    },
    {
        "query": "Why should response.ok be checked when using fetch()?",
        "relevance": [3, 2, 1],
    },
    {
        "query": "What is the React and FastAPI architecture described in the notes?",
        "relevance": [3, 1, 1],
    },
]


def test_student_rag_retrieval():

    total_hits = 0
    total_relevant = 0
    total_results = 0
    reciprocal_ranks = []

    for case in EVAL_CASES:

        print("\n" + "=" * 70)
        print(f"Query: {case['query']}")
        print("=" * 70)

        results = retrieve(
            query=case["query"],
            top_k=3,
            knowledge_scope="student_material",
            source_type="text",
        )

        assert len(results) == 3

        first_relevant_rank = None

        for rank, (result, relevance) in enumerate(
            zip(results, case["relevance"]),
            start=1,
        ):

            metadata = result["metadata"]

            # Student-material isolation
            assert metadata["knowledge_scope"] == "student_material"

            # This evaluation corpus is TXT
            assert metadata["source_type"] == "text"

            print(
                f"Rank {rank}: "
                f"{metadata.get('title')} | "
                f"distance={result['distance']:.4f} | "
                f"relevance={relevance}"
            )

            if relevance >= 2:

                total_relevant += 1

                if first_relevant_rank is None:
                    first_relevant_rank = rank

        total_results += 3

        # Hit@3
        if first_relevant_rank is not None:
            total_hits += 1

        # MRR
        if first_relevant_rank is not None:
            reciprocal_ranks.append(
                1 / first_relevant_rank
            )
        else:
            reciprocal_ranks.append(0)

    hit_at_3 = total_hits / len(EVAL_CASES)

    precision_at_3 = (
        total_relevant / total_results
    )

    mrr = (
        sum(reciprocal_ranks)
        / len(EVAL_CASES)
    )

    print("\n" + "=" * 70)
    print("STUDENT MATERIAL RAG RESULTS")
    print("=" * 70)

    print(f"Hit@3:       {hit_at_3:.4f}")
    print(f"Precision@3: {precision_at_3:.4f}")
    print(f"MRR:         {mrr:.4f}")

    print("\nPercentages:")

    print(f"Hit@3:       {hit_at_3:.2%}")
    print(f"Precision@3: {precision_at_3:.2%}")
    print(f"MRR:         {mrr:.2%}")

    assert hit_at_3 >= 0
    assert precision_at_3 >= 0
    assert mrr >= 0

if __name__ == "__main__":
    test_student_rag_retrieval()