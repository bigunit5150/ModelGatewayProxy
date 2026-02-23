"""Sentence-transformer embedding model for semantic cache similarity matching.

The :class:`EmbeddingModel` wraps ``sentence-transformers`` so that:

* The heavy model load happens lazily on the first call (not at import time).
* CPU-bound encoding runs in a thread-pool executor so the async event loop
  is never blocked.
* Every encode call is instrumented with an OpenTelemetry span and a
  Prometheus histogram.

Cosine similarity is computed as the dot product of two unit-norm vectors,
which is valid because :meth:`EmbeddingModel.encode` always returns
``normalize_embeddings=True`` output.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import numpy as np
import structlog
from opentelemetry import trace
from prometheus_client import Histogram

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_log = structlog.get_logger(__name__)
_tracer = trace.get_tracer(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

_EMBEDDING_DURATION = Histogram(
    "llm_embedding_generation_duration_seconds",
    "Time to generate a single text embedding (thread-pool execution)",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5],
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# EmbeddingModel
# ---------------------------------------------------------------------------


class EmbeddingModel:
    """Async wrapper around a sentence-transformers model.

    The underlying :class:`~sentence_transformers.SentenceTransformer` is
    loaded lazily on the first :meth:`encode` call and then reused for the
    lifetime of the instance.

    Args:
        model_name: HuggingFace model identifier.  Defaults to
                    ``"all-MiniLM-L6-v2"`` — 384 dimensions, ~22 MB,
                    fast inference (typically < 10 ms on CPU).
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> SentenceTransformer:
        """Load the model on first call (thread-safe: GIL protects the check)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            _log.info("embedding.model_loading", model=self._model_name)
            self._model = SentenceTransformer(self._model_name)
            _log.info("embedding.model_ready", model=self._model_name)
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def encode(self, text: str) -> np.ndarray:
        """Return a normalised ``float32`` embedding vector for *text*.

        Encoding runs in a thread-pool executor so the event loop is never
        blocked.  The returned vector has unit L2 norm, making cosine
        similarity equivalent to the dot product.

        Args:
            text: Input string to embed.

        Returns:
            1-D ``numpy.ndarray`` of shape ``(384,)`` and dtype ``float32``.
        """
        model = self._load()

        with _tracer.start_as_current_span("embedding.encode") as span:
            span.set_attribute("embedding.model", self._model_name)
            span.set_attribute("embedding.text_length", len(text))

            start = time.monotonic()
            vec: np.ndarray = await asyncio.to_thread(
                model.encode,
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            duration = time.monotonic() - start
            span.set_attribute("embedding.duration_ms", round(duration * 1000, 2))

        _EMBEDDING_DURATION.observe(duration)
        _log.debug(
            "embedding.encoded",
            model=self._model_name,
            text_length=len(text),
            duration_ms=round(duration * 1000, 2),
        )
        return vec.astype(np.float32)

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Return cosine similarity in ``[-1, 1]`` between two unit-norm vectors.

        Both inputs must already be unit-norm (as returned by :meth:`encode`).
        Cosine similarity then reduces to the dot product, which avoids a
        division and is numerically stable.

        Args:
            vec1: First normalised embedding vector.
            vec2: Second normalised embedding vector.

        Returns:
            Similarity score.  For unit vectors this is in ``[-1.0, 1.0]``;
            identical sentences will be very close to ``1.0``.
        """
        return float(np.dot(vec1, vec2))
