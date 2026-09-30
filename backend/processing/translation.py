import os
from google import genai
from dotenv import load_dotenv

load_dotenv()


client = genai.Client(api_key=os.getenv("LLM_API_KEY"))

def translate_to_eng(text:str) -> str:

    if not text.strip():
        raise ValueError("Text to translate cannot be empty")

    prompt = f"""
Translate the following educational transcript into natural, clear English.

The source may be Hindi, Hinglish, or another language and may contain
English technical terms, programming keywords, library names, product names,
code, URLs, and other technical terminology.

Requirements:
- Preserve the speaker's intended meaning.
- Translate the natural-language portions into English.
- Preserve technical terms such as Python, LangChain, RAG, API, FastAPI,
  embeddings, vector database, etc. exactly when appropriate.
- Preserve library names, framework names, function names, variable names,
  code, URLs, and proper nouns exactly.
- Do not translate technical terminology merely because it appears inside
  a Hindi or mixed-language sentence.
- Correct obvious caption/transcription errors using surrounding context.
- Do not translate word-for-word when that produces unnatural English.
- Do not summarize, shorten, or omit meaningful content.
- Do not add information that is not supported by the transcript.
- Preserve examples and technical explanations accurately.
- Return ONLY the English translation.

Source transcript:
{text}
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents = prompt
    )

    translated_text= response.text.strip()

    if not translated_text: 
        raise ValueError("Translation returned empty text")

    return translated_text


