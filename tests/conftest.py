"""Shared fixtures for vault-semantic-search tests."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_VAULT = FIXTURES_DIR / "sample_vault"


@pytest.fixture
def sample_vault() -> Path:
    """Return the path to the sample vault fixture directory."""
    return SAMPLE_VAULT


@pytest.fixture
def tmp_vault(tmp_path: Path) -> Path:
    """Return a temporary vault with a .obsidian directory."""
    obsidian = tmp_path / ".obsidian"
    obsidian.mkdir()
    return tmp_path
