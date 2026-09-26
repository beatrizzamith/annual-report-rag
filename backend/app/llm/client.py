"""One thin client for chat (structured output) and embeddings.

Supports OpenAI and Azure OpenAI behind the same interface (`LLMClient`), so
the rest of the codebase never branches on provider. Retries with backoff on
rate limits and transient errors; raises typed errors on auth failure.
"""

import logging
import time
from functools import partial
from typing import Any, Protocol, TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    AzureOpenAI,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel

from app.config import Settings
from app.core.errors import LLMAuthFailedError, LLMRateLimitedError, LLMUnavailableError

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

_MAX_RETRIES = 3
_BASE_BACKOFF_SECONDS = 1.5


class LLMClient(Protocol):
    """Interface for chat (structured output) and embedding calls.

    Implemented by `OpenAICompatibleClient` for real use and by
    `app.llm.fakes.FakeLLM` for tests, so the rest of the codebase never
    depends on a concrete provider.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embeds a batch of texts.

        Args:
            texts: The texts to embed.

        Returns:
            One embedding vector per input text, in the same order.
        """
        ...

    def complete_structured(
        self,
        system: str,
        user: str,
        schema: type[SchemaT],
        temperature: float = 0,
    ) -> SchemaT:
        """Runs a chat completion constrained to a Pydantic schema.

        Args:
            system: The system prompt.
            user: The user prompt.
            schema: The Pydantic model the response must conform to.
            temperature: Sampling temperature; 0 for repeatable output.

        Returns:
            An instance of `schema` populated from the model's response.
        """
        ...


def _call_with_retries(operation: partial, what: str) -> Any:
    """Runs `operation` with retries and typed error translation.

    Args:
        operation: A zero-argument callable (typically a `functools.partial`
            over an SDK method) that performs one API call.
        what: A short description of the call, used in error messages.

    Returns:
        Whatever `operation()` returns, from its first successful attempt.

    Raises:
        LLMAuthFailedError: The provider rejected the credentials.
        LLMRateLimitedError: The provider kept rate-limiting every attempt.
        LLMUnavailableError: The provider was unreachable or erroring on
            every attempt.
    """
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return operation()
        except AuthenticationError as exc:
            raise LLMAuthFailedError(f"{what}: authentication failed") from exc
        except RateLimitError as exc:
            last_exc = exc
            time.sleep(_BASE_BACKOFF_SECONDS * attempt)
        except (APITimeoutError, APIConnectionError, APIStatusError) as exc:
            last_exc = exc
            time.sleep(_BASE_BACKOFF_SECONDS * attempt)
    if isinstance(last_exc, RateLimitError):
        raise LLMRateLimitedError(f"{what}: rate limited after {_MAX_RETRIES} attempts")
    raise LLMUnavailableError(f"{what}: unavailable after {_MAX_RETRIES} attempts: {last_exc}")


class OpenAICompatibleClient:
    """Real `LLMClient` implementation, backed by the `openai` SDK.

    Uses `OpenAI` for the `openai` provider and `AzureOpenAI` for `azure`.
    For Azure, `chat_model`/`embedding_model` are deployment names.
    """

    def __init__(self, settings: Settings):
        """Builds the underlying SDK client for the configured provider.

        Args:
            settings: Application settings, including which provider to use
                and its credentials.

        Raises:
            LLMAuthFailedError: The configured provider is missing required
                credentials.
        """
        self._settings = settings
        if settings.llm_provider == "azure":
            if not (settings.azure_openai_api_key and settings.azure_openai_endpoint):
                raise LLMAuthFailedError("Azure OpenAI is not configured")
            self._client = AzureOpenAI(
                api_key=settings.azure_openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
            )
        else:
            if not settings.openai_api_key:
                raise LLMAuthFailedError("OpenAI is not configured")
            self._client = OpenAI(api_key=settings.openai_api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embeds a batch of texts via the configured embedding model.

        Args:
            texts: The texts to embed.

        Returns:
            One embedding vector per input text, in the same order. An
            empty list if `texts` is empty.

        Raises:
            LLMAuthFailedError: The provider rejected the credentials.
            LLMRateLimitedError: The provider kept rate-limiting every retry.
            LLMUnavailableError: The provider was unreachable on every retry.
        """
        if not texts:
            return []
        operation = partial(
            self._client.embeddings.create,
            model=self._settings.embedding_model,
            input=texts,
            timeout=30,
        )
        response = _call_with_retries(operation, what="embedding")
        return [item.embedding for item in response.data]

    def complete_structured(
        self,
        system: str,
        user: str,
        schema: type[SchemaT],
        temperature: float = 0,
    ) -> SchemaT:
        """Runs a chat completion constrained to a Pydantic schema.

        Args:
            system: The system prompt.
            user: The user prompt.
            schema: The Pydantic model the response must conform to.
            temperature: Sampling temperature; 0 for repeatable output.

        Returns:
            An instance of `schema` populated from the model's response.

        Raises:
            LLMAuthFailedError: The provider rejected the credentials.
            LLMRateLimitedError: The provider kept rate-limiting every retry.
            LLMUnavailableError: The provider was unreachable on every retry,
                or returned no parsable output.
        """
        operation = partial(
            self._client.chat.completions.parse,
            model=self._settings.chat_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=schema,
            temperature=temperature,
            timeout=60,
        )
        completion = _call_with_retries(operation, what="chat completion")
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise LLMUnavailableError("chat completion: model refused to produce output")
        return parsed


def build_llm_client(settings: Settings) -> LLMClient:
    """Builds the real `LLMClient` for the configured provider.

    Args:
        settings: Application settings, including which provider to use and
            its credentials.

    Returns:
        An `OpenAICompatibleClient` for the configured provider.

    Raises:
        LLMAuthFailedError: The configured provider is missing required
            credentials.
    """
    return OpenAICompatibleClient(settings)
