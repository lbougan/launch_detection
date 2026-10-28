from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.synthetic import PlantedSite, make_synthetic_composite


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="needs --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def synthetic_composite(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, PlantedSite]:
    """One 1024x1024 composite with a planted site, shared by the session."""
    path = tmp_path_factory.mktemp("composite") / "synthetic_composite.tif"
    return make_synthetic_composite(path)


@pytest.fixture
def small_composite(tmp_path: Path) -> tuple[Path, PlantedSite]:
    """A 320x320 composite for fast unit tests."""
    return make_synthetic_composite(tmp_path / "small.tif", height=320, width=320, site_size_px=24)
