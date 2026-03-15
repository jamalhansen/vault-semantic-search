"""Defaults, paths, and configuration for vault-semantic-search."""

import os
from pathlib import Path


# --- Embedding model ---
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_BATCH_SIZE = 32

# --- ChromaDB ---
def get_db_path() -> Path:
    """XDG-compliant path for the ChromaDB database."""
    xdg_data = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
    return Path(xdg_data) / "vsearch" / "chromadb"


COLLECTION_NAME = "vault"

# --- Ollama ---
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embed"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"
OLLAMA_TIMEOUT = 120.0

# --- Chunking ---
MAX_CHUNK_TOKENS = 1000       # approx tokens; we use word count as proxy (1 token ≈ 0.75 words)
MIN_CHUNK_TOKENS = 50
WORDS_PER_TOKEN = 0.75        # conversion factor: words / WORDS_PER_TOKEN = approx tokens

# --- Vault ---
VSEARCH_VAULT_ENV = "VSEARCH_VAULT"

# Files and dirs to skip by default (no .vsearchignore present)
DEFAULT_SKIP_DIRS = {".obsidian", "_templates", "_template"}
DEFAULT_SKIP_PREFIXES = (".",)  # hidden files/dirs
MIN_FILE_CHARS = 50             # skip files smaller than this

VSEARCH_IGNORE_FILE = ".vsearchignore"

# --- Search ---
DEFAULT_TOP_K = 5
SNIPPET_LENGTH = 200  # chars to show in result snippet
