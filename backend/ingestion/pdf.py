from pathlib import Path

from pypdf import PdfReader

from .base import Document


def extract_pdf(file_path: str) -> Document:

    path = Path(file_path)

    reader = PdfReader(path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        if text.strip():

            # Normalize whitespace produced by PDF text extraction.
            # pypdf can place newlines between individual text elements.
            text = " ".join(text.split())

            pages.append({
                "page": page_number,
                "text": text
            })

    if not pages:
        raise ValueError(
            "No readable text found. Try another file."
        )

    full_text = "\n".join(
        page["text"] for page in pages
    )

    return Document(
        source_type="pdf",
        title=path.stem,
        text=full_text,
        metadata={
            "filename": path.name,
            "page_count": len(reader.pages),
            "pages": pages
        },
    )