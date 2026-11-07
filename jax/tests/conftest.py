"""Shared fixtures for the aevplig_jax test suite."""

import importlib.util

import numpy as np
import pytest

TORCH_AVAILABLE = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "torch_geometric")
)
_SKIP_NO_TORCH = "needs torch and torch_geometric"


def pytest_collection_modifyitems(config, items):
    """Skip tests marked ``torch`` when torch or torch_geometric is not installed."""
    if TORCH_AVAILABLE:
        return
    skip = pytest.mark.skip(reason=_SKIP_NO_TORCH)
    for item in items:
        if item.get_closest_marker("torch") is not None:
            item.add_marker(skip)


@pytest.fixture
def torch_available():
    """Skip the requesting test when torch or torch_geometric is not installed."""
    if not TORCH_AVAILABLE:
        pytest.skip(_SKIP_NO_TORCH)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def point_set(rng):
    """30 ligand points and 200 pocket points in a 12 A box.

    Pocket species are uniform over 1..22, then 10% of them are set to 0 (untyped).
    Returns ``(lig_xyz, env_xyz, env_species)``.
    """
    lig_xyz = rng.uniform(0.0, 12.0, size=(30, 3))
    env_xyz = rng.uniform(0.0, 12.0, size=(200, 3))
    env_species = rng.integers(1, 23, size=200)
    env_species[rng.choice(200, size=20, replace=False)] = 0
    return lig_xyz, env_xyz, env_species
