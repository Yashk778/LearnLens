from rag.vectorstore import delete_student_material
from rag.integrate_student import index_student_document
from ingestion.text import extract_text
from ingestion.pdf import extract_pdf
from ingestion.base import Document


print("Deleting old student chunks...")
delete_student_material()

print("Indexing TXT...")
txt_doc = extract_text("data/sample/React — Short Notes.txt")
print("TXT chunks:", index_student_document(txt_doc))

print("Indexing PDF...")
pdf_doc = extract_pdf("data/sample/React — Notes.pdf")
print("PDF chunks:", index_student_document(pdf_doc))

print("Indexing YouTube translation...")
youtube_doc = Document(
    source_type="youtube",
    title="AI Complete Crash Course for Beginners in Hindi",
    text=open(
        "data/sample/youtube_ai_crash_course_en.txt",
        encoding="utf-8"
    ).read(),
    url="https://youtu.be/SeuS84YeJVc",
    metadata={
        "video_id": "SeuS84YeJVc",
        "caption_source": "youtube",
        "language": "hi",
        "is_generated": True,
        "needs_translation": True,
        "translated_to": "en",
        "original_language": "hi",
    },
)

print("YouTube chunks:", index_student_document(youtube_doc))

print("Student material re-indexing complete.")