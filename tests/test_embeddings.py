"""Tests for vsearch.embeddings."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from vsearch.embeddings import OllamaError, embed_texts, _embed_batch


class TestEmbedTexts:
    def test_returns_empty_for_empty_input(self):
        result = embed_texts([])
        assert result == []

    def test_batches_correctly(self):
        """Verify that large input is split into batches."""
        fake_embedding = [0.1] * 768
        texts = [f"text {i}" for i in range(70)]

        call_args_list = []

        def fake_embed_batch(batch, model):
            call_args_list.append(len(batch))
            return [fake_embedding] * len(batch)

        with patch("vsearch.embeddings._embed_batch", side_effect=fake_embed_batch):
            result = embed_texts(texts, model="nomic-embed-text", batch_size=32)

        assert len(result) == 70
        # Should have called with 32, 32, 6
        assert call_args_list == [32, 32, 6]

    def test_single_text_returns_single_embedding(self):
        fake_embedding = [0.5] * 768
        with patch("vsearch.embeddings._embed_batch", return_value=[fake_embedding]):
            result = embed_texts(["hello"], batch_size=32)
        assert result == [fake_embedding]


class TestEmbedBatch:
    def _mock_response(self, embeddings: list, status_code: int = 200):
        mock = MagicMock()
        mock.status_code = status_code
        mock.json.return_value = {"embeddings": embeddings}
        mock.raise_for_status = MagicMock()
        if status_code >= 400:
            mock.raise_for_status.side_effect = httpx.HTTPStatusError(
                "error", request=MagicMock(), response=mock
            )
        return mock

    def test_successful_embed(self):
        fake_embeds = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        with patch("httpx.post", return_value=self._mock_response(fake_embeds)):
            result = _embed_batch(["text1", "text2"], "nomic-embed-text")
        assert result == fake_embeds

    def test_connect_error_raises_ollama_error(self):
        with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(OllamaError, match="Ollama is not running"):
                _embed_batch(["text"], "nomic-embed-text")

    def test_timeout_raises_ollama_error(self):
        with patch("httpx.post", side_effect=httpx.TimeoutException("timeout")):
            with pytest.raises(OllamaError, match="timed out"):
                _embed_batch(["text"], "nomic-embed-text")

    def test_404_raises_model_not_found(self):
        mock = self._mock_response([], status_code=404)
        with patch("httpx.post", return_value=mock):
            with pytest.raises(OllamaError, match="not found in Ollama"):
                _embed_batch(["text"], "bad-model")

    def test_500_raises_ollama_error(self):
        mock = self._mock_response([], status_code=500)
        with patch("httpx.post", return_value=mock):
            with pytest.raises(OllamaError, match="Ollama returned an error"):
                _embed_batch(["text"], "nomic-embed-text")

    def test_passes_model_to_ollama(self):
        fake_embeds = [[0.1]]
        captured = {}

        def fake_post(url, json, timeout):
            captured["json"] = json
            mock = MagicMock()
            mock.status_code = 200
            mock.json.return_value = {"embeddings": fake_embeds}
            mock.raise_for_status = MagicMock()
            return mock

        with patch("httpx.post", side_effect=fake_post):
            _embed_batch(["hello"], "mxbai-embed-large")

        assert captured["json"]["model"] == "mxbai-embed-large"
        assert captured["json"]["input"] == ["hello"]


@pytest.mark.integration
class TestEmbedTextsIntegration:
    """These tests require Ollama to be running with nomic-embed-text pulled."""

    def test_returns_vectors_of_expected_dimension(self):
        result = embed_texts(["hello world"], model="nomic-embed-text")
        assert len(result) == 1
        assert len(result[0]) == 768

    def test_batch_produces_same_count_as_input(self):
        texts = [f"sentence number {i}" for i in range(10)]
        result = embed_texts(texts, model="nomic-embed-text", batch_size=4)
        assert len(result) == len(texts)

    def test_different_texts_have_different_embeddings(self):
        result = embed_texts(
            ["SQL NULL handling", "sourdough bread recipe"],
            model="nomic-embed-text",
        )
        assert result[0] != result[1]
