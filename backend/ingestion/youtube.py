from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
)

from .base import Document


def extract_video_id(url: str) -> str:
    """Extract the YouTube video ID from a standard or shortened URL."""

    parsed_url = urlparse(url)

    if parsed_url.hostname in {"www.youtube.com", "youtube.com"}:
        # Parse the query string into a dictionary, look up the 'v' key,
        # and safely grab the first item in the list or None if missing.
        return parse_qs(parsed_url.query).get("v", [None])[0]

    if parsed_url.hostname == "youtu.be":
        # For shortened links, the video ID is the URL path itself.
        return parsed_url.path.strip("/")

    raise ValueError("Invalid YouTube URL.")


def get_best_transcript(video_id: str) -> dict:
    """Select the best available YouTube caption track."""

    try:
        transcript_list = YouTubeTranscriptApi().list(video_id)

    except (NoTranscriptFound, TranscriptsDisabled):
        raise ValueError(
            "No captions available for this YouTube video."
        )

    english_codes = [
        "en",
        "en-IN",
        "en-GB",
        "en-US",
        "en-AU",
        "en-CA",
    ]

    # Prefer manually created English captions.
    for code in english_codes:
        try:
            transcript = (
                transcript_list
                .find_manually_created_transcript([code])
            )

            snippets = transcript.fetch()

            text = " ".join(
                snippet.text.strip()
                for snippet in snippets
                if snippet.text.strip()
            )

            return {
                "text": text,
                "language": transcript.language_code,
                "is_generated": False,
                "needs_translation": False,
            }

        except Exception:
            continue

    # Use auto-generated English if manual English is unavailable.
    for code in english_codes:
        try:
            transcript = (
                transcript_list
                .find_generated_transcript([code])
            )

            snippets = transcript.fetch()

            text = " ".join(
                snippet.text.strip()
                for snippet in snippets
                if snippet.text.strip()
            )

            return {
                "text": text,
                "language": transcript.language_code,
                "is_generated": True,
                "needs_translation": False,
            }

        except Exception:
            continue

    # If English is unavailable, use another available language.
    transcripts = list(transcript_list)

    if not transcripts:
        raise ValueError(
            "No captions available for this YouTube video."
        )

    # Prefer manually created captions.
    manual_transcripts = [
        transcript
        for transcript in transcripts
        if not transcript.is_generated
    ]

    transcript = (
        manual_transcripts[0]
        if manual_transcripts
        else transcripts[0]
    )

    snippets = transcript.fetch()

    text = " ".join(
        snippet.text.strip()
        for snippet in snippets
        if snippet.text.strip()
    )

    return {
        "text": text,
        "language": transcript.language_code,
        "is_generated": transcript.is_generated,
        "needs_translation": True,
    }


def extract_vid_text(url: str) -> Document:
    """Extract caption text from a YouTube video."""

    video_id = extract_video_id(url)

    if not video_id:
        raise ValueError(
            "Could not find YouTube video ID."
        )

    transcript = get_best_transcript(video_id)

    if not transcript["text"]:
        raise ValueError(
            "No readable captions found for this video."
        )

    return Document(
        source_type="youtube",
        title=f"YouTube video {video_id}",
        text=transcript["text"],
        url=url,
        metadata={
            "video_id": video_id,
            "caption_source": "youtube",
            "language": transcript["language"],
            "is_generated": transcript["is_generated"],
            "needs_translation": transcript["needs_translation"],
        },
    )