from uuid import uuid4

from backend.ingestion.base import Document
from backend.processing.cleaning import clean_text
from backend.rag.chunking import chunk_text
from backend.rag.embeddings import embed_texts
from backend.rag.vectorstore import add_docs


def index_student_document(document: Document) -> int:
    """
    Index a student-provided document into the shared LearnLens
    ChromaDB collection.
    """

    if not document.text.strip():
        raise ValueError("Student document cannot be empty.")

    cleaned_text = clean_text(document.text)
    chunks = chunk_text(cleaned_text)

    if not chunks:
        raise ValueError("Student document produced no chunks.")

    embeddings = embed_texts(chunks)

    metadatas = []

    for index, _ in enumerate(chunks, start=1):
        metadatas.append(
            {
                "knowledge_scope": "student_material",
                "source_type": document.source_type,
                "source_name": document.metadata.get(
                    "filename",
                    document.title,
                ),
                "title": document.title,
                "url": document.url or "",
                "chunk_index": index,
            }
        )

    ids = [str(uuid4()) for _ in chunks]

    add_docs(
        chunks=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    return len(chunks)