import httpx
from bs4 import BeautifulSoup

from backend.ingestion.base import Document


WIKIPEDIA_API = "https://en.wikipedia.org/w/rest.php/v1"


def fetch_wikipedia_page(
    title: str,
    topic: str,
) -> Document:
    """Fetch a Wikipedia article through the Wikimedia REST API."""

    if not title.strip():
        raise ValueError("Wikipedia title cannot be empty.")

    # Convert the article title into URL format
    encoded_title = title.replace(" ", "_")

    url = f"{WIKIPEDIA_API}/page/{encoded_title}/with_html"

    # Identify the application making the API request
    # Use your public GitHub repo URL (preferred by Wikimedia)
    headers = {
        "User-Agent": (
            "LearnLens/1.0 "
            "(https://github.com/Yashk778/LearnLens; educational AI tutor)"
        )
    }

    # Request the article from Wikimedia
    response = httpx.get(
        url,
        headers=headers,
        timeout=20,
        follow_redirects=True,
    )

    response.raise_for_status()

    # Convert the API response from JSON into a Python dictionary
    data = response.json()

    html = data.get("html")

    if not html:
        raise ValueError(
            f"No readable content found for Wikipedia page: {title}"
        )

    # Convert the returned HTML into readable plain text
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")

    # Remove empty lines and unnecessary whitespace
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]
    text = "\n".join(lines)

    if not text:
        raise ValueError(
            f"No readable text found for Wikipedia page: {title}"
        )

    # Preserve Wikipedia licensing information
    license_info = data.get("license", {})

    # Preserve revision information for source tracking
    latest = data.get("latest", {})
    revision_id = latest.get("id")
    revision_timestamp = latest.get("timestamp")

    # Use the URL provided by Wikimedia when available
    page_url = data.get(
        "html_url",
        f"https://en.wikipedia.org/wiki/{encoded_title}"
    )

    # Convert the Wikipedia article into LearnLens's standard Document format
    return Document(
        source_type="wikipedia",
        title=data.get("title", title),
        text=text,
        url=page_url,
        metadata={
            "source_name": "wikipedia",
            "topic": topic,
            "verified_source": True,
            "license": license_info.get("title"),
            "license_url": license_info.get("url"),
            "revision_id": revision_id,
            "revision_timestamp": revision_timestamp,
        },
    )