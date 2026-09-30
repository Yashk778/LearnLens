import re


def clean_text(text: str) -> str:
    """
    Normalize extracted or translated learning material
    before chunking and embedding.
    """

    if not text.strip():
        raise ValueError("Text to clean cannot be empty.")

    # Normalize line breaks
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Collapse repeated whitespace while preserving paragraphs
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize repeated punctuation
    text = re.sub(r"[.]{4,}", "...", text)
    text = re.sub(r"[!?]{3,}", "!!", text)

    return text.strip()