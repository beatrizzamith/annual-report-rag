"""Application configuration, loaded from environment variables and `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    llm_provider: Literal["openai", "azure"] = "openai"

    openai_api_key: str | None = None

    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str = "2024-10-21"

    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    data_dir: Path = Path("./data")
    max_upload_mb: int = 150
    context_token_budget: int = 5000

    log_level: str = "INFO"

    @property
    def pdfs_dir(self) -> Path:
        """Returns the directory used for storing original PDFs.

        Returns:
            The path `data_dir / "pdfs"`.
        """
        return self.data_dir / "pdfs"

    @property
    def db_path(self) -> Path:
        """Returns the SQLite database path.

        Returns:
            The path `data_dir / "app.db"`.
        """
        return self.data_dir / "app.db"

    @property
    def embedding_dim(self) -> int:
        """Returns the vector dimension for the configured embedding model.

        Returns:
            The known dimension for `embedding_model`, defaulting to 1536
            when the model name is not recognised.
        """
        return {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }.get(self.embedding_model, 1536)

    @property
    def llm_configured(self) -> bool:
        """Returns whether the configured LLM provider has valid credentials.

        Returns:
            True if the configured provider's required credentials are set.
        """
        if self.llm_provider == "openai":
            return bool(self.openai_api_key)
        return bool(self.azure_openai_api_key and self.azure_openai_endpoint)


@lru_cache
def get_settings() -> Settings:
    """Builds and caches the application's settings for the process lifetime.

    Returns:
        The application's settings, read from environment variables/`.env`
        on first call and cached thereafter.
    """
    return Settings()
