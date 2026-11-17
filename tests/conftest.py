from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from antifraude import data, model
from antifraude.api import create_app
from antifraude.config import Settings


@pytest.fixture(scope="session")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, model.Metrics]:
    model_dir = tmp_path_factory.mktemp("models")
    pipe, metrics = model.train(data.generate(n=10_000, seed=7))
    model.save(pipe, metrics, model_dir)
    return model_dir, metrics


@pytest.fixture
def client(trained: tuple[Path, model.Metrics], tmp_path: Path) -> Iterator[TestClient]:
    settings = Settings(database_url=f"sqlite:///{tmp_path}/test.db", model_dir=trained[0])
    with TestClient(create_app(settings)) as c:
        yield c
