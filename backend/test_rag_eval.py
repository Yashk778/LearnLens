from backend.rag.retrieval import retrieve


EVAL_CASES = [
    {
        "query": "What is machine learning?",
        "source_type": "wikipedia",
        "expected_title": "Machine learning",
        "relevance": [1, 2, 2],
    },
    {
        "query": "What is a Python function?",
        "source_type": "official_docs",
        "expected_title": "Python Tutorial",
        "relevance": [2, 1, 3],
    },
    {
        "query": "What is a NumPy array?",
        "source_type": "official_docs",
        "expected_title": "NumPy Quickstart",
        "relevance": [3, 2, 3],
    },
    {
        "query": "What is a pandas DataFrame?",
        "source_type": "official_docs",
        "expected_title": "10 Minutes to pandas",
        "relevance": [3, 1, 2],
    },
    {
        "query": "What is a neural network?",
        "source_type": "wikipedia",
        "expected_title": "Machine learning",
        "relevance": [3, 2, 3],
    },
    {
        "query": "What is a PyTorch tensor?",
        "source_type": "official_docs",
        "expected_title": "Learn the Basics",
        "relevance": [3, 2, 1],
    },
    {
        "query": "What is a transformer model?",
        "source_type": "official_docs",
        "expected_title": "Transformers Documentation",
        "relevance": [2, 3, 2],
    },
    {
        "query": "What is LangGraph?",
        "source_type": "official_docs",
        "expected_title": "LangGraph Overview",
        "relevance": [3, 3, 2],
    },
]


def test_rag_retrieval_baseline():

    total_relevant = 0
    total_retrieved = 0
    hit_count = 0
    reciprocal_rank_sum = 0

    for case in EVAL_CASES:

        results = retrieve(
            case["query"],
            top_k=3,
            knowledge_scope="built_in",
            source_type=case["source_type"],
        )

        assert results
        assert len(results) == 3

        relevance = case["relevance"]

        # --------------------------------
        # Source isolation check
        # --------------------------------

        for result in results:
            assert (
                result["metadata"]["source_type"]
                == case["source_type"]
            )

        # --------------------------------
        # Hit@3
        # --------------------------------

        relevant_results = [
            score >= 2
            for score in relevance
        ]

        if any(relevant_results):
            hit_count += 1

        # --------------------------------
        # Precision@3
        # --------------------------------

        total_relevant += sum(relevant_results)
        total_retrieved += len(results)

        # --------------------------------
        # MRR
        # --------------------------------

        first_relevant_rank = None

        for rank, score in enumerate(relevance, start=1):

            if score >= 2:
                first_relevant_rank = rank
                break

        if first_relevant_rank is not None:
            reciprocal_rank_sum += 1 / first_relevant_rank

        # --------------------------------
        # Display
        # --------------------------------

        print("\n" + "=" * 70)
        print(f"Query: {case['query']}")

        for rank, (result, score) in enumerate(
            zip(results, relevance),
            start=1
        ):

            metadata = result["metadata"]

            print(
                f"Rank {rank}: "
                f"{metadata['title']} | "
                f"distance={result['distance']:.4f} | "
                f"relevance={score}"
            )

    # --------------------------------
    # Final metrics
    # --------------------------------

    hit_at_3 = hit_count / len(EVAL_CASES)

    precision_at_3 = (
        total_relevant / total_retrieved
    )

    mrr = (
        reciprocal_rank_sum / len(EVAL_CASES)
    )

    print("\n" + "=" * 70)
    print("RAG BASELINE RESULTS")
    print("=" * 70)

    print(f"Hit@3:       {hit_at_3:.4f}")
    print(f"Precision@3:  {precision_at_3:.4f}")
    print(f"MRR:          {mrr:.4f}")

    print("\nPercentages:")
    print(f"Hit@3:       {hit_at_3 * 100:.2f}%")
    print(f"Precision@3: {precision_at_3 * 100:.2f}%")
    print(f"MRR:         {mrr * 100:.2f}%")