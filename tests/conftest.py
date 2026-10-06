import pytest

from rotation_fit import load_config, load_hr_data, train
from rotation_fit.synthetic import make_synthetic, synthetic_config_overrides

FAST = {"split_ratios": [0.3], "seeds": [1, 2, 3], "scalers": ["standard"],
        "models": ["dummy", "logreg", "rf"], "importance_repeats": 3}


@pytest.fixture(scope="session")
def synth_paths(tmp_path_factory):
    return make_synthetic(tmp_path_factory.mktemp("synthetic"), n_employees=600, seed=0)


@pytest.fixture(scope="session")
def cfg(synth_paths, tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    ov = synthetic_config_overrides(synth_paths)
    ov["paths"].update({"outputs": str(out / "outputs"), "report": str(out / "report")})
    ov["experiment"] = FAST
    return load_config(overrides=ov)


@pytest.fixture(scope="session")
def hr(cfg):
    return load_hr_data(cfg)


@pytest.fixture(scope="session")
def result(cfg, hr):
    return train(cfg, hr=hr)
