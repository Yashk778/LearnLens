from uuid import uuid4

from backend.sources.curated_sources import CURATED_SOURCES
from backend.sources.official_docs import fetch_official_doc
from backend.sources.wikipedia import fetch_wikipedia_page
from backend.processing.cleaning import clean_text
from backend.rag.chunking import chunk_text
from backend.rag.embeddings import embed_texts
from backend.rag.vectorstore import add_docs


def index_curated_sources():
    for topic, sources in CURATED_SOURCES.items():

        for source in sources:

            print(f"\nIndexing: {source['title']}")

            if source["type"] == "official_docs":

                document = fetch_official_doc(
                    url=source["url"],
                    title=source["title"],
                    topic=topic,
                    source_name=source["source"],
                )

            elif source["type"] == "wikipedia":

                document = fetch_wikipedia_page(
                    title=source["title"],
                    topic=topic,
                )

            else:
                raise ValueError(
                    f"Unsupported source type: {source['type']}"
                )

            cleaned_text = clean_text(
                document.text
            )

            chunks = chunk_text(
                cleaned_text
            )

            embeddings = embed_texts(
                chunks
            )

            metadatas = [
                {
                    "knowledge_scope": "built_in",
                    "source_type": document.source_type,
                    "source_name": source["source"],
                    "title": document.title,
                    "url": document.url,
                    "topic": topic,
                    "license": document.metadata.get(
                        "license",
                        "",
                    ),
                    "license_url": document.metadata.get(
                        "license_url",
                        "",
                    ),
                    "revision_id": str(
                        document.metadata.get(
                            "revision_id",
                            "",
                        )
                    ),
                    "revision_timestamp": document.metadata.get(
                        "revision_timestamp",
                        "",
                    ),
                }
                for _ in chunks
            ]

            ids = [
                str(uuid4())
                for _ in chunks
            ]

            add_docs(
                chunks=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )

            print(
                f"Indexed {len(chunks)} chunks"
            )


if __name__ == "__main__":
    index_curated_sources()