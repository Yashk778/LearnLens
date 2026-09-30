from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of text chunks into embedding vectors.
    """

    if not texts:
        raise ValueError("Texts to embed cannot be empty.")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )

    return embeddings.tolist()


def get_embedding_dimension() -> int:
    """Return the dimensionality of the embedding vectors."""

    return model.get_embedding_dimension()