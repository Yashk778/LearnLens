from .embeddings import embed_texts
from .vectorstore import collection


def retrieve(
    query: str,
    top_k: int = 3
) -> list[dict]:

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    retrieved_chunks = []

    for document, distance, metadata in zip(
        documents,
        distances,
        metadatas
    ):
        retrieved_chunks.append({
            "text": document,
            "distance": distance,
            "metadata": metadata
        })

    return retrieved_chunks