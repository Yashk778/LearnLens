import chromadb

COLLECTION_NAME= "LEARNLENS"

client = chromadb.PersistentClient(
    path = 'chroma_db'
)


collection =  client.get_or_create_collection(
    name = COLLECTION_NAME
)

def add_docs(chunks:list[str],embeddings:list[list[float]],metadatas:list[dict],ids:list[str]) -> None:

    """
    Store chunks, embeddings, and metadata in ChromaDB.
    """

    if not chunks :
        raise ValueError("Chunks cannot be empty.")

    if not (
        len(chunks)
        == len(embeddings)
        == len(metadatas)
        == len(ids)
    ):
        raise ValueError(
            "Chunks, embeddings, metadata, and IDs "
            "must have the same length."
        )


    collection.add(
        ids = ids,
        documents = chunks,
        embeddings  = embeddings,
        metadatas=metadatas
    )

def get_collection_count() -> int:
    """Return the number of stored documents."""

    return collection.count()



def delete_student_material() -> None:
    """Delete only student-material chunks."""
    
    student_ids = collection.get(
        where={"knowledge_scope": "student_material"},
        include=[]
    )["ids"]

    if student_ids:
        collection.delete(ids=student_ids)