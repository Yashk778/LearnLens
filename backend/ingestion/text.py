from pathlib import Path
from .base import Document


def extract_text(file_path :str) -> Document:

    path = Path(file_path)

    text = path.read_text(encoding='utf-8').strip()

    if not text :
        raise ValueError (
            "No readable text found. Try another file."
        )

    return Document(
            source_type='text',
            title=path.stem,
            text = text,
            metadata={
                "filename":path.name
            },
        )