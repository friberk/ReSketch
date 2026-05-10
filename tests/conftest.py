from collections.abc import Callable
from pathlib import Path

import pytest

from resketch.config import AppConfig, load_config


@pytest.fixture
def load_fixture_config(tmp_path: Path) -> Callable[..., AppConfig]:
    def _load(provider: str = "fixture") -> AppConfig:
        return load_config(
            overrides={
                "retrieval.provider": provider,
                "retrieval.cache_path": str(tmp_path / "cache.json"),
                "cegis.interactive": False,
            }
        )

    return _load
