"""Loads prompt text files from the repo's `prompts/` directory."""

from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


@lru_cache
def load_prompt(name: str) -> str:
    """Reads and caches one prompt file's contents.

    Args:
        name: The prompt file's name, e.g. "extract_fte.md".

    Returns:
        The prompt file's contents.

    Raises:
        FileNotFoundError: No such file exists under `PROMPTS_DIR`.
    """
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")
