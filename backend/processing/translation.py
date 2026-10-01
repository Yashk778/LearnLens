from backend.rag.chunking import chunk_text
from backend.llm.client import generate_text


def translate_chunk(text: str) -> str:
    if not text.strip():
        raise ValueError("Text to translate cannot be empty")

    prompt = f"""
Translate the following educational transcript into natural, clear English.

The source may be Hindi, Hinglish, or another language and may contain
English technical terms, programming keywords, library names, product names,
code, URLs, and other technical terminology.

Requirements:
- Preserve the speaker's intended meaning.
- Translate natural-language portions into English.
- Preserve technical terms such as Python, LangChain, RAG, API, FastAPI,
  embeddings, vector database, etc. exactly when appropriate.
- Preserve library names, framework names, function names, variable names,
  code, URLs, and proper nouns exactly.
- Correct obvious caption/transcription errors using surrounding context.
- Do not summarize, shorten, or omit meaningful content.
- Do not add information that is not supported by the transcript.
- Preserve examples and technical explanations accurately.
- Return ONLY the English translation.

Source transcript:
{text}
"""

    translated_text = generate_text(prompt)

    if not translated_text:
        raise ValueError("Translation returned empty text")

    return translated_text


def translate_to_eng(text: str) -> str:
    if not text.strip():
        raise ValueError("Text to translate cannot be empty")

    chunks = chunk_text(
        text,
        chunk_size=6000,
        chunk_overlap=300,
    )

    translated_chunks = [
        translate_chunk(chunk)
        for chunk in chunks
    ]

    translated_text = "\n\n".join(translated_chunks)

    if not translated_text.strip():
        raise ValueError("Translation returned empty text")

    return translated_text