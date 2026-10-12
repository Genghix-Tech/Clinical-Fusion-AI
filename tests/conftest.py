import pytest

from clinicalfusion.config import Config
from clinicalfusion.experiments.runner import run_experiments


@pytest.fixture(scope="session")
def tiny_cfg():
    return Config.fast(n_patients=500, epochs_img=2, epochs_ts=3, epochs_fusion=3, mc_samples=4, n_bootstrap=30, n_explain=4)


@pytest.fixture(scope="session")
def tiny_run(tiny_cfg):
    """One shared end-to-end run (the slow part) reused by several tests."""
    return run_experiments(tiny_cfg)
