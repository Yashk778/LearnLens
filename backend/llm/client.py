import os

from dotenv import load_dotenv
from google import genai
from openai import OpenAI

load_dotenv()

GEMINI_API_KEY = os.getenv("LLM_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

gemini_client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY
    else None
)

openrouter_client = (
    OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    if OPENROUTER_API_KEY
    else None
)


def generate_text(prompt: str) -> str:
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

  
    # PRIMARY: GEMINI
   

    if gemini_client:
        try:
            response = gemini_client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
            )

            text = response.text.strip()

            if text:
                return text

        except Exception as error:
            print(f"Gemini failed. Trying OpenRouter: {error}")

    
    # FALLBACK: OPENROUTER
    

    if openrouter_client:
        try:
            response = openrouter_client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                timeout=30,
            )

            text = response.choices[0].message.content.strip()

            if text:
                return text

        except Exception as error:
            print(f"OpenRouter failed: {error}")

    raise RuntimeError("No LLM provider is currently available.")