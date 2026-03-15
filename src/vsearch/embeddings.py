"""Ollama embedding client for vault-semantic-search."""

from __future__ import annotations

import httpx

from vsearch.config import (
    EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    OLLAMA_EMBED_URL,
    OLLAMA_TIMEOUT,
)


class OllamaError(RuntimeError):
    """Raised when Ollama is unavailable or returns an error."""


def embed_texts(
    texts: list[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> list[list[float]]:
    """Embed a list of texts via Ollama, batching requests.

    Returns a list of embedding vectors (one per input text).

    Raises OllamaError with a clear message if Ollama is not running or
    the model is not available.
    """
    if not texts:
        return []

    embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings.extend(_embed_batch(batch, model))
    return embeddings


def _embed_batch(texts: list[str], model: str) -> list[list[float]]:
    try:
        response = httpx.post(
            OLLAMA_EMBED_URL,
            json={"model": model, "input": texts},
            timeout=OLLAMA_TIMEOUT,
        )
    except httpx.ConnectError:
        raise OllamaError(
            "Ollama is not running. Start it with `ollama serve` or check that it's installed."
        )
    except httpx.TimeoutException:
        raise OllamaError(
            f"Ollama request timed out after {OLLAMA_TIMEOUT}s. "
            "Try reducing batch size or using a lighter model."
        )

    if response.status_code == 404:
        raise OllamaError(
            f"Model '{model}' not found in Ollama. "
            f"Pull it with: ollama pull {model}"
        )

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise OllamaError(f"Ollama returned an error: {e}") from e

    data = response.json()
    return data["embeddings"]
