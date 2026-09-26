"""Tests for where the application's data and settings are found."""

from pathlib import Path

import pytest

from app.config import BACKEND_DIR, Settings


def test_default_data_dir_is_inside_the_backend_folder_whatever_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DATA_DIR", raising=False)

    settings = Settings(_env_file=None)

    assert settings.data_dir == BACKEND_DIR / "data"
    assert settings.db_path == BACKEND_DIR / "data" / "app.db"


def test_relative_data_dir_is_resolved_against_the_backend_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)

    settings = Settings(data_dir=Path("./data"), _env_file=None)

    assert settings.data_dir == BACKEND_DIR / "data"


def test_absolute_data_dir_is_left_unchanged(tmp_path: Path):
    assert Settings(data_dir=tmp_path, _env_file=None).data_dir == tmp_path
