from pathlib import Path
from .base import Document
from pypdf import PdfReader

def extract_pdf(file_path :str) -> Document:

    path = Path(file_path)

    reader  = PdfReader(path)

    pages = []

    for page_number ,page  in enumerate(reader.pages,start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page":page_number,
                "text":text.strip()
            })

    if not pages :
        raise ValueError (
            "No readable text found. Try another file."
        )
    full_text = "\n".join(
        page["text"] for page in pages
    )

    return Document(
        source_type='pdf',
        title=path.stem,
        text = full_text,
        metadata={
            "filename":path.name,
            "page_count":len(reader.pages),
            "pages":pages
        },
    )


