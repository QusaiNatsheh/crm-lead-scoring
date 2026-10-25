import pytest
from crm_score_ml.generator import Generator, GeneratorConfig, save
from crm_score_ml.train import train


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    path = tmp_path_factory.mktemp("data")
    cfg = GeneratorConfig(seed=3, leads_per_day=4.0)
    save(Generator(cfg).generate(), path, cfg)
    return path


@pytest.fixture(scope="session")
def trained(dataset, tmp_path_factory):
    import os

    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{tmp_path_factory.mktemp('mlflow')}/m.db"
    out = tmp_path_factory.mktemp("model")
    meta = train(dataset, out)
    return out, meta
