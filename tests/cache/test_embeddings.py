"""Tests for EmbeddingModel (cache/embeddings.py)."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from llmgateway.cache.embeddings import EmbeddingModel

# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_unit_vectors_returns_one(self) -> None:
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert EmbeddingModel.cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors_returns_zero(self) -> None:
        vec1 = np.array([1.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0], dtype=np.float32)
        assert EmbeddingModel.cosine_similarity(vec1, vec2) == pytest.approx(0.0)

    def test_opposite_unit_vectors_returns_minus_one(self) -> None:
        vec1 = np.array([1.0, 0.0], dtype=np.float32)
        vec2 = np.array([-1.0, 0.0], dtype=np.float32)
        assert EmbeddingModel.cosine_similarity(vec1, vec2) == pytest.approx(-1.0)

    def test_returns_python_float(self) -> None:
        vec = np.array([0.6, 0.8], dtype=np.float32)
        result = EmbeddingModel.cosine_similarity(vec, vec)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Model loading (_load)
# ---------------------------------------------------------------------------


class TestLoad:
    def test_model_is_none_before_first_call(self) -> None:
        model = EmbeddingModel()
        assert model._model is None

    def test_custom_model_name_is_stored(self) -> None:
        model = EmbeddingModel("paraphrase-MiniLM-L6-v2")
        assert model._model_name == "paraphrase-MiniLM-L6-v2"

    def test_load_constructs_sentence_transformer(self) -> None:
        model = EmbeddingModel()
        mock_st = MagicMock()
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_st) as mock_cls:
            result = model._load()
        assert result is mock_st
        assert model._model is mock_st
        mock_cls.assert_called_once_with("all-MiniLM-L6-v2")

    def test_load_caches_model_on_subsequent_calls(self) -> None:
        model = EmbeddingModel()
        mock_st = MagicMock()
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_st) as mock_cls:
            first = model._load()
            second = model._load()
        assert first is second
        mock_cls.assert_called_once()


# ---------------------------------------------------------------------------
# encode()
# ---------------------------------------------------------------------------


class TestEncode:
    async def test_returns_float32_array(self) -> None:
        model = EmbeddingModel()
        expected = np.array([0.1, 0.2, 0.3], dtype=np.float64)  # float64 input

        with patch("sentence_transformers.SentenceTransformer", return_value=MagicMock()):
            with patch("asyncio.to_thread", new=AsyncMock(return_value=expected)):
                result = await model.encode("test text")

        assert result.dtype == np.float32

    async def test_encodes_via_thread_pool(self) -> None:
        model = EmbeddingModel()
        mock_vec = np.ones(384, dtype=np.float32)

        with patch("sentence_transformers.SentenceTransformer", return_value=MagicMock()):
            with patch("asyncio.to_thread", new=AsyncMock(return_value=mock_vec)) as mock_thread:
                await model.encode("hello world")

        mock_thread.assert_called_once()

    async def test_model_loaded_on_first_encode(self) -> None:
        model = EmbeddingModel()
        assert model._model is None
        mock_st = MagicMock()
        mock_vec = np.ones(384, dtype=np.float32)

        with patch("sentence_transformers.SentenceTransformer", return_value=mock_st):
            with patch("asyncio.to_thread", new=AsyncMock(return_value=mock_vec)):
                await model.encode("hello")

        assert model._model is mock_st
