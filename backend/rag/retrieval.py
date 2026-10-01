from .embeddings import embed_texts
from .vectorstore import collection


def retrieve(
    query: str,
    top_k: int = 3,
    knowledge_scope: str | None = None,
    source_type: str | None = None,
) -> list[dict]:

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    query_embedding = embed_texts([query])[0]

    # Build Chroma metadata filter
    where = None

    if knowledge_scope and source_type:
        where = {
            "$and": [
                {"knowledge_scope": knowledge_scope},
                {"source_type": source_type},
            ]
        }

    elif knowledge_scope:
        where = {
            "knowledge_scope": knowledge_scope
        }

    elif source_type:
        where = {
            "source_type": source_type
        }

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
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
        retrieved_chunks.append(
            {
                "text": document,
                "distance": distance,
                "metadata": metadata,
            }
        )

    return retrieved_chunks

