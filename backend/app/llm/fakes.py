"""Fakes for `LLMClient`, used by tests so the whole pipeline runs offline.

`FakeEmbedder` produces deterministic, semantically-meaningless-but-stable
vectors from a hashed bag of words, which is enough to test that retrieval
plumbing (fusion, scoping, budget) works without a real embedding model.

`FakeLLM.complete_structured` returns pre-scripted responses from a queue, so
a test can assert exactly what the pipeline does with a given model output,
including "the model returned this fabricated quote" cases.
"""

from collections import deque
from collections.abc import Callable
from typing import TypeVar

import numpy as np
from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class FakeEmbedder:
    """Deterministic, hash-based stand-in for a real embedding model."""

    def __init__(self, dim: int = 32):
        """Sets the vector dimension this embedder will produce.

        Args:
            dim: The dimension of vectors this embedder produces.
        """
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embeds a batch of texts deterministically.

        Args:
            texts: The texts to embed.

        Returns:
            One deterministic, L2-normalised vector per input text.
        """
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        """Embeds a single text deterministically.

        Args:
            text: The text to embed.

        Returns:
            An L2-normalised vector built from a hashed bag of `text`'s
            lower-cased words.
        """
        vec = np.zeros(self.dim, dtype=np.float32)
        for word in text.lower().split():
            idx = hash(word) % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()


class FakeLLMError(RuntimeError):
    """Raised when `FakeLLM` is used incorrectly by a test (empty queue, or
    a queued response of the wrong schema)."""


class FakeLLM:
    """Queue-driven fake `LLMClient`. `queue_response` schedules the next
    return value(s) for `complete_structured`; a bare exception instance can
    be queued to simulate a failed call (e.g. rewrite timeout)."""

    def __init__(self, embedder: FakeEmbedder | None = None):
        """Sets up an empty response queue and call log.

        Args:
            embedder: The embedder to delegate `embed` to. Defaults to a new
                `FakeEmbedder`.
        """
        self.embedder = embedder or FakeEmbedder()
        self._queue: deque[BaseModel | Exception | Callable[[], BaseModel]] = deque()
        self.calls: list[tuple[str, str]] = []

    def queue_response(self, response: BaseModel | Exception | Callable[[], BaseModel]) -> None:
        """Schedules the next value `complete_structured` will return or raise.

        Args:
            response: A Pydantic model instance to return, an exception
                instance to raise, or a zero-argument callable that produces
                either when invoked.
        """
        self._queue.append(response)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embeds a batch of texts by delegating to `self.embedder`.

        Args:
            texts: The texts to embed.

        Returns:
            One deterministic vector per input text.
        """
        return self.embedder.embed(texts)

    def complete_structured(
        self,
        system: str,
        user: str,
        schema: type[SchemaT],
        temperature: float = 0,
    ) -> SchemaT:
        """Returns (or raises) the next response scheduled by `queue_response`.

        Args:
            system: The system prompt (recorded in `self.calls`, not used).
            user: The user prompt (recorded in `self.calls`, not used).
            schema: The Pydantic model the queued response must be an
                instance of.
            temperature: Unused; accepted to match `LLMClient`.

        Returns:
            The next response queued via `queue_response`.

        Raises:
            FakeLLMError: No response was queued, or the queued response is
                not an instance of `schema`.
            Exception: Whatever exception was queued via `queue_response`,
                re-raised as-is.
        """
        self.calls.append((system, user))
        if not self._queue:
            raise FakeLLMError("FakeLLM: no scripted response queued")
        item = self._queue.popleft()
        if isinstance(item, Exception):
            raise item
        result = item() if callable(item) else item
        if not isinstance(result, schema):
            raise FakeLLMError(f"FakeLLM: queued {type(result)} but caller expects {schema}")
        return result
