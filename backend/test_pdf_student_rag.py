from rag.retrieval import retrieve


EVAL_CASES = [
    {"query": "What is React and what is Vite used for?", "relevance": [3, 1, 1]},
    {"query": "What are React components?", "relevance": [2, 3, 1]},
    {"query": "What are props in React?", "relevance": [1, 3, 1]},
    {"query": "What does map() do in React?", "relevance": [3, 1, 1]},
    {"query": "What is the difference between props and state?", "relevance": [2, 1, 3]},
    {"query": "What is useState used for?", "relevance": [1, 3, 1]},
    {"query": "What is useEffect used for?", "relevance": [3, 1, 1]},
    {"query": "How does React communicate with FastAPI?", "relevance": [3, 1, 2]},
    {"query": "Why should response.ok be checked when using fetch()?", "relevance": [3, 2, 1]},
    {"query": "What is the React and FastAPI architecture described in the notes?", "relevance": [3, 1, 1]},
]


def test_pdf_student_rag():

    hits = 0
    relevant_results = 0
    reciprocal_ranks = []

    for case in EVAL_CASES:

        results = retrieve(
            query=case["query"],
            top_k=3,
            knowledge_scope="student_material",
            source_type="pdf",
        )

        assert len(results) == 3

        for result in results:
            metadata = result["metadata"]

            assert metadata["knowledge_scope"] == "student_material"
            assert metadata["source_type"] == "pdf"

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

    print("\nPDF STUDENT RAG RESULTS")
    print("=" * 50)
    print(f"Hit@3:       {hit_at_3:.2%}")
    print(f"Precision@3: {precision_at_3:.2%}")
    print(f"MRR:         {mrr:.2%}")
    print("Source isolation: PASSED")


if __name__ == "__main__":
    test_pdf_student_rag()