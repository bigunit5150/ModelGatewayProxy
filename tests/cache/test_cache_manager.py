"""Tests for CacheManager (exact-match and semantic cache)."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock

import numpy as np
import pytest

from llmgateway.cache.base import CacheEntry
from llmgateway.cache.cache_manager import CacheManager, _extract_user_text
from llmgateway.cache.embeddings import EmbeddingModel
from llmgateway.providers import CompletionRequest

# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------


def _req(
    *,
    temperature: float = 0.0,
    messages: list[dict] | None = None,
    model: str = "gpt-4o",
    max_tokens: int | None = None,
) -> CompletionRequest:
    if messages is None:
        messages = [{"role": "user", "content": "Hello, world!"}]
    return CompletionRequest(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _backend(*, entry: CacheEntry | None = None) -> AsyncMock:
    mock = AsyncMock()
    mock.get.return_value = entry
    mock.set.return_value = None
    return mock


def _redis() -> AsyncMock:
    mock = AsyncMock()
    mock.zrange.return_value = []
    mock.mget.return_value = []
    mock.zadd.return_value = None
    mock.zcard.return_value = 1
    mock.set.return_value = None
    mock.get.return_value = None
    mock.zpopmin.return_value = []
    mock.delete.return_value = None
    return mock


def _embedding(*, vec: np.ndarray | None = None) -> AsyncMock:
    if vec is None:
        vec = np.ones(384, dtype=np.float32)
    mock = AsyncMock(spec=EmbeddingModel)
    mock.encode.return_value = vec
    return mock


def _entry(value: str = '{"id": "chatcmpl-1"}') -> CacheEntry:
    return CacheEntry(
        key="test-key",
        value=value,
        created_at=time.time(),
        ttl=3600,
        metadata={"model": "gpt-4o"},
    )


_SAMPLE: dict = {
    "id": "chatcmpl-xyz",
    "object": "chat.completion",
    "created": 1000,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hi!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}


# ---------------------------------------------------------------------------
# _extract_user_text
# ---------------------------------------------------------------------------


class TestExtractUserText:
    def test_returns_content_of_last_user_message(self) -> None:
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        assert _extract_user_text(msgs) == "Hello!"

    def test_returns_last_when_multiple_user_messages(self) -> None:
        msgs = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Reply"},
            {"role": "user", "content": "Second"},
        ]
        assert _extract_user_text(msgs) == "Second"

    def test_returns_empty_string_when_no_user_message(self) -> None:
        msgs = [{"role": "assistant", "content": "Hello!"}]
        assert _extract_user_text(msgs) == ""

    def test_returns_empty_string_for_empty_list(self) -> None:
        assert _extract_user_text([]) == ""


# ---------------------------------------------------------------------------
# generate_cache_key
# ---------------------------------------------------------------------------


class TestGenerateCacheKey:
    def test_deterministic_for_same_inputs(self) -> None:
        mgr = CacheManager(backend=_backend())
        k1 = mgr.generate_cache_key("gpt-4o", [{"role": "user", "content": "Hi"}], 0.0, None)
        k2 = mgr.generate_cache_key("gpt-4o", [{"role": "user", "content": "Hi"}], 0.0, None)
        assert k1 == k2

    def test_different_model_gives_different_key(self) -> None:
        mgr = CacheManager(backend=_backend())
        k1 = mgr.generate_cache_key("gpt-4o", [{"role": "user", "content": "Hi"}], 0.0, None)
        k2 = mgr.generate_cache_key("gpt-4o-mini", [{"role": "user", "content": "Hi"}], 0.0, None)
        assert k1 != k2

    def test_key_is_64_char_hex(self) -> None:
        mgr = CacheManager(backend=_backend())
        key = mgr.generate_cache_key("gpt-4o", [{"role": "user", "content": "Hi"}], 0.0, None)
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


# ---------------------------------------------------------------------------
# get_cached_response (exact match)
# ---------------------------------------------------------------------------


class TestGetCachedResponse:
    async def test_skips_nonzero_temperature(self) -> None:
        mgr = CacheManager(backend=_backend())
        assert await mgr.get_cached_response(_req(temperature=0.5)) is None

    async def test_returns_none_on_backend_miss(self) -> None:
        mgr = CacheManager(backend=_backend(entry=None))
        assert await mgr.get_cached_response(_req()) is None

    async def test_returns_parsed_dict_on_hit(self) -> None:
        entry = _entry(json.dumps(_SAMPLE))
        mgr = CacheManager(backend=_backend(entry=entry))
        result = await mgr.get_cached_response(_req())
        assert result is not None
        assert result["id"] == "chatcmpl-xyz"

    async def test_returns_none_for_corrupt_cached_json(self) -> None:
        entry = _entry("not-valid-json{{{")
        mgr = CacheManager(backend=_backend(entry=entry))
        assert await mgr.get_cached_response(_req()) is None


# ---------------------------------------------------------------------------
# cache_response (exact match)
# ---------------------------------------------------------------------------


class TestCacheResponse:
    async def test_skips_nonzero_temperature(self) -> None:
        be = _backend()
        mgr = CacheManager(backend=be)
        await mgr.cache_response(_req(temperature=0.7), _SAMPLE)
        be.set.assert_not_called()

    async def test_stores_entry_for_zero_temperature(self) -> None:
        be = _backend()
        mgr = CacheManager(backend=be)
        await mgr.cache_response(_req(), _SAMPLE)
        be.set.assert_called_once()
        stored: CacheEntry = be.set.call_args[0][0]
        assert json.loads(stored.value) == _SAMPLE

    async def test_custom_ttl_applied(self) -> None:
        be = _backend()
        mgr = CacheManager(backend=be, default_ttl=3600)
        await mgr.cache_response(_req(), _SAMPLE, ttl=60)
        stored: CacheEntry = be.set.call_args[0][0]
        assert stored.ttl == 60


# ---------------------------------------------------------------------------
# semantic_enabled property
# ---------------------------------------------------------------------------


class TestSemanticEnabled:
    def test_true_when_both_redis_and_embedding_provided(self) -> None:
        mgr = CacheManager(backend=_backend(), redis_client=_redis(), embedding_model=_embedding())
        assert mgr.semantic_enabled is True

    def test_false_when_redis_is_none(self) -> None:
        mgr = CacheManager(backend=_backend(), redis_client=None, embedding_model=_embedding())
        assert mgr.semantic_enabled is False

    def test_false_when_embedding_is_none(self) -> None:
        mgr = CacheManager(backend=_backend(), redis_client=_redis(), embedding_model=None)
        assert mgr.semantic_enabled is False


# ---------------------------------------------------------------------------
# get_semantic_match
# ---------------------------------------------------------------------------


class TestGetSemanticMatch:
    async def test_returns_none_when_semantic_disabled(self) -> None:
        mgr = CacheManager(backend=_backend())
        assert await mgr.get_semantic_match(_req()) is None

    async def test_returns_none_for_nonzero_temperature(self) -> None:
        mgr = CacheManager(backend=_backend(), redis_client=_redis(), embedding_model=_embedding())
        assert await mgr.get_semantic_match(_req(temperature=0.5)) is None

    async def test_returns_none_when_no_user_message(self) -> None:
        mgr = CacheManager(backend=_backend(), redis_client=_redis(), embedding_model=_embedding())
        request = _req(messages=[{"role": "system", "content": "You are helpful."}])
        assert await mgr.get_semantic_match(request) is None

    async def test_returns_none_when_index_is_empty(self) -> None:
        r = _redis()
        r.zrange.return_value = []
        mgr = CacheManager(backend=_backend(), redis_client=r, embedding_model=_embedding())
        assert await mgr.get_semantic_match(_req()) is None

    async def test_returns_none_when_similarity_below_threshold(self) -> None:
        query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        stored_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # orthogonal → sim=0

        r = _redis()
        r.zrange.return_value = ["entry-1"]
        r.mget.return_value = [json.dumps(stored_vec.tolist())]

        mgr = CacheManager(
            backend=_backend(),
            redis_client=r,
            embedding_model=_embedding(vec=query_vec),
            semantic_threshold=0.95,
        )
        assert await mgr.get_semantic_match(_req()) is None

    async def test_returns_response_when_similarity_above_threshold(self) -> None:
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # sim with itself = 1.0

        r = _redis()
        r.zrange.return_value = ["entry-1"]
        r.mget.return_value = [json.dumps(vec.tolist())]
        r.get.return_value = json.dumps(_SAMPLE)

        mgr = CacheManager(
            backend=_backend(),
            redis_client=r,
            embedding_model=_embedding(vec=vec),
            semantic_threshold=0.95,
        )
        result = await mgr.get_semantic_match(_req())

        assert result is not None
        response, similarity = result
        assert similarity == pytest.approx(1.0)
        assert response["id"] == "chatcmpl-xyz"

    async def test_returns_none_when_response_key_expired(self) -> None:
        vec = np.array([1.0, 0.0], dtype=np.float32)

        r = _redis()
        r.zrange.return_value = ["entry-1"]
        r.mget.return_value = [json.dumps(vec.tolist())]
        r.get.return_value = None  # response key has expired

        mgr = CacheManager(
            backend=_backend(),
            redis_client=r,
            embedding_model=_embedding(vec=vec),
            semantic_threshold=0.95,
        )
        assert await mgr.get_semantic_match(_req()) is None

    async def test_skips_expired_embedding_entries(self) -> None:
        vec = np.array([1.0, 0.0], dtype=np.float32)

        r = _redis()
        r.zrange.return_value = ["expired-entry", "live-entry"]
        r.mget.return_value = [None, json.dumps(vec.tolist())]  # first expired
        r.get.return_value = json.dumps(_SAMPLE)

        mgr = CacheManager(
            backend=_backend(),
            redis_client=r,
            embedding_model=_embedding(vec=vec),
            semantic_threshold=0.95,
        )
        result = await mgr.get_semantic_match(_req())
        assert result is not None  # live-entry matched

    async def test_encode_error_is_fail_open(self) -> None:
        mock_emb = AsyncMock(spec=EmbeddingModel)
        mock_emb.encode.side_effect = RuntimeError("model error")

        mgr = CacheManager(
            backend=_backend(),
            redis_client=_redis(),
            embedding_model=mock_emb,
        )
        assert await mgr.get_semantic_match(_req()) is None


# ---------------------------------------------------------------------------
# cache_with_embedding
# ---------------------------------------------------------------------------


class TestCacheWithEmbedding:
    async def test_noop_when_semantic_disabled(self) -> None:
        mgr = CacheManager(backend=_backend())
        await mgr.cache_with_embedding(_req(), _SAMPLE)  # should not raise

    async def test_noop_for_nonzero_temperature(self) -> None:
        r = _redis()
        mgr = CacheManager(backend=_backend(), redis_client=r, embedding_model=_embedding())
        await mgr.cache_with_embedding(_req(temperature=0.7), _SAMPLE)
        r.set.assert_not_called()

    async def test_noop_when_no_user_message(self) -> None:
        r = _redis()
        mgr = CacheManager(backend=_backend(), redis_client=r, embedding_model=_embedding())
        await mgr.cache_with_embedding(
            _req(messages=[{"role": "system", "content": "You are helpful."}]),
            _SAMPLE,
        )
        r.set.assert_not_called()

    async def test_stores_embedding_and_response_in_redis(self) -> None:
        r = _redis()
        r.zcard.return_value = 1  # below limit

        mgr = CacheManager(
            backend=_backend(),
            redis_client=r,
            embedding_model=_embedding(),
            semantic_max_entries=100,
        )
        await mgr.cache_with_embedding(_req(), _SAMPLE)

        assert r.set.call_count == 2  # embedding key + response key
        r.zadd.assert_called_once()

    async def test_evicts_oldest_entry_when_index_full(self) -> None:
        r = _redis()
        r.zcard.return_value = 11  # exceeds max of 10
        r.zpopmin.return_value = [("old-entry-id", 1000.0)]

        mgr = CacheManager(
            backend=_backend(),
            redis_client=r,
            embedding_model=_embedding(),
            semantic_max_entries=10,
        )
        await mgr.cache_with_embedding(_req(), _SAMPLE)

        r.zpopmin.assert_called_once()
        r.delete.assert_called_once()

    async def test_encode_error_is_fail_open(self) -> None:
        mock_emb = AsyncMock(spec=EmbeddingModel)
        mock_emb.encode.side_effect = RuntimeError("GPU OOM")

        mgr = CacheManager(
            backend=_backend(),
            redis_client=_redis(),
            embedding_model=mock_emb,
        )
        await mgr.cache_with_embedding(_req(), _SAMPLE)  # should not raise
