import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.fixtures.build_fixture import build_fixture_pdf  # noqa: E402


@pytest.fixture
def fixture_pdf_path(tmp_path: Path) -> Path:
    return build_fixture_pdf(tmp_path / "fixture.pdf")


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "data"
    directory.mkdir()
    return directory
