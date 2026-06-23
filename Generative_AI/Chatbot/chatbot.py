"""
chatbot.py
==========
Groq backend wrapper for the Deepak AI Assistant.

Responsibilities:
- Create/maintain a Groq client (local .env key OR user-supplied key)
- Expose available models
- Provide a streaming response generator
- Provide a non-streaming response helper (kept for backward compatibility)
- Centralised, friendly error handling (invalid key, rate limit, network, etc.)
"""

from __future__ import annotations

import os
import time
from typing import Generator, List, Dict, Optional

from dotenv import load_dotenv
from groq import Groq
from groq import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)

load_dotenv()


# =========================
# AVAILABLE MODELS
# =========================

AVAILABLE_MODELS: Dict[str, str] = {
    "Llama 3.3 70B (Versatile)": "llama-3.3-70b-versatile",
    "Llama 3.1 8B (Instant)": "llama-3.1-8b-instant",
    "Llama 4 Scout 17B": "meta-llama/llama-4-scout-17b-16e-instruct",
    "Llama 4 Maverick 17B": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "Gemma 2 9B": "gemma2-9b-it",
    "DeepSeek R1 Distill (Llama 70B)": "deepseek-r1-distill-llama-70b",
}

DEFAULT_MODEL_LABEL = "Llama 3.3 70B (Versatile)"


class ChatbotError(Exception):
    """Raised for any chatbot-related failure with a user-friendly message."""


def _get_client(user_api_key: Optional[str] = None) -> Groq:
    """
    Build a Groq client.

    Priority:
    1. Local .env GROQ_API_KEY (if present)
    2. User-supplied API key (sidebar input)

    Raises:
        ChatbotError: if no key is available at all.
    """
    api_key = os.getenv("GROQ_API_KEY") or user_api_key

    if not api_key:
        raise ChatbotError(
            "No Groq API key found. Please enter your API key in the sidebar."
        )

    try:
        return Groq(api_key=api_key)
    except Exception as exc:  # pragma: no cover - defensive
        raise ChatbotError(f"Could not initialise Groq client: {exc}") from exc


def _clean_messages_for_api(messages: List[Dict]) -> List[Dict]:
    """
    Strip any UI-only keys (e.g. 'timestamp') before sending to the Groq API.
    Groq only accepts 'role' and 'content'.
    """
    cleaned = []
    for msg in messages:
        cleaned.append({"role": msg["role"], "content": msg["content"]})
    return cleaned


def stream_response(
    messages: List[Dict],
    api_key: Optional[str] = None,
    model: str = AVAILABLE_MODELS[DEFAULT_MODEL_LABEL],
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> Generator[str, None, None]:
    """
    Stream a chat completion from Groq, chunk by chunk.

    Yields:
        str: incremental text chunks.

    Raises:
        ChatbotError: friendly, normalised error on any failure.
    """
    client = _get_client(api_key)
    payload = _clean_messages_for_api(messages)

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=payload,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    except AuthenticationError as exc:
        raise ChatbotError(
            "Invalid Groq API key. Please check the key in the sidebar and try again."
        ) from exc

    except RateLimitError as exc:
        raise ChatbotError(
            "Groq rate limit reached. Please wait a few seconds and try again."
        ) from exc

    except APIConnectionError as exc:
        raise ChatbotError(
            "Network error: could not reach Groq's servers. Check your internet connection."
        ) from exc

    except APIStatusError as exc:
        raise ChatbotError(
            f"Groq API returned an error (status {exc.status_code}). {exc.message}"
        ) from exc

    except Exception as exc:  # pragma: no cover - defensive catch-all
        raise ChatbotError(f"Unexpected error while talking to Groq: {exc}") from exc


def get_response(
    messages: List[Dict],
    api_key: Optional[str] = None,
    model: str = AVAILABLE_MODELS[DEFAULT_MODEL_LABEL],
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """
    Non-streaming helper retained for backward compatibility.
    Internally consumes the streaming generator and returns the full text.

    Returns:
        str: the full assistant response.

    Raises:
        ChatbotError: friendly, normalised error on any failure.
    """
    chunks = []
    for chunk in stream_response(
        messages,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    ):
        chunks.append(chunk)
    return "".join(chunks)


def time_request(func, *args, **kwargs):
    """
    Utility to measure elapsed wall-clock time (seconds) for any callable.

    Returns:
        tuple: (result, elapsed_seconds)
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed