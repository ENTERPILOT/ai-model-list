"""Helpers for extracting official Google Speech pricing needed for Vertex AI entries."""

from __future__ import annotations

from typing import Any


CHIRP_MARKER = "chirp\u00a0(Speech-to-Text V2 only)"
CHIRP_PRICE_MARKER = "$0.003 / 1 minute, per 1 month / account"


def build_google_speech_models_snapshot(html_text: str, source_url: str) -> list[dict[str, Any]]:
    if CHIRP_MARKER not in html_text or CHIRP_PRICE_MARKER not in html_text:
        raise ValueError("unable to locate Chirp pricing in Google Speech docs")

    return [
        {
            "id": "vertex_ai",
            "pricing_urls": [source_url],
            "models": [
                {
                    "id": "chirp",
                    "name": "Chirp",
                    "description": "Speech-to-Text V2 standard recognition model.",
                    "mode": "audio_transcription",
                    "match": {"equals": "chirp"},
                    "prices": {
                        "per_second_input": 0.003 / 60,
                    },
                }
            ],
        }
    ]
