

import httpx 
from bs4 import BeautifulSoup
from backend.ingestion.base import Document

def fetch_official_doc(url:str,title:str,topic:str,source_name:str) -> Document:
    #Fetch a curated official documentation page and convert it into the standard Document format.
    

    if not url.strip():
        raise ValueError("Document cannot be empty.")

    response = httpx.get(
        url,
        timeout=20,
        follow_redirects=True
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text,'html.parser')

    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()

    main_content = soup.find("main")

    if main_content:
        text = main_content.get_text(separator="\n")
    else:
        text = soup.get_text(separator="\n")

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    text = "\n".join(lines)

    if not text:
        raise ValueError(
            "No readable documentation content found."
        )

    return Document(
        source_type="official_docs",
        title=title,
        text=text,
        url=str(response.url),
        metadata={
            "source_name": source_name,
            "topic": topic,
            "verified_source": True,
        }
    )


