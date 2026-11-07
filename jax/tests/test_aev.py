"""Radial AEV: numpy brute-force reference, cKDTree neighbour lists, JAX implementation."""

from pathlib import Path

import jax
import numpy as np
import pytest

from aevplig_jax.aev import neighbour_lists, radial_aev_jax, radial_aev_numpy
from aevplig_jax.constants import N_SPECIES, species_table

# Values copied verbatim from generate_bindingnet_graphs.py (ANI-2x radial coefficients).
RC = 5.1
ETA = 19.7
RS = np.array([0.80, 1.07, 1.34, 1.61, 1.88, 2.14, 2.41, 2.68,
               2.95, 3.22, 3.49, 3.76, 4.03, 4.29, 4.56, 4.83])
KEYS_CSV = Path(__file__).resolve().parents[2] / "data" / "PDB_Atom_Keys.csv"


def _pair(distance, species):
    """One ligand atom at the origin and one pocket atom on the x axis."""
    return np.zeros((1, 3)), np.array([[distance, 0.0, 0.0]]), np.array([species])


def _aev_jax(lig_xyz, env_xyz, env_species, k=None):
    nbr, mask = neighbour_lists(lig_xyz, env_xyz, k=k)
    return np.asarray(radial_aev_jax(lig_xyz, env_xyz, env_species, nbr, mask))


def test_numpy_matches_brute_force_single_pair():
    aev = radial_aev_numpy(*_pair(2.0, 7))
    assert aev.shape == (1, 352)
    assert aev.dtype == np.float64
    expected = 0.25 * np.exp(-ETA * (2.0 - RS) ** 2) * (0.5 * np.cos(np.pi * 2.0 / RC) + 0.5)
    cols = (7 - 1) * 16 + np.arange(16)
    np.testing.assert_allclose(aev[0, cols], expected, rtol=1e-12, atol=0)
    assert np.all(np.delete(aev[0], cols) == 0.0)


@pytest.mark.parametrize("distance", [5.1, 6.0])
def test_beyond_cutoff_is_zero(distance):
    pair = _pair(distance, 7)
    assert np.all(radial_aev_numpy(*pair) == 0.0)
    assert np.all(_aev_jax(*pair) == 0.0)


def test_species_zero_contributes_nothing():
    lig = np.zeros((1, 3))
    env = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    with_untyped = np.array([0, 3])
    typed_only = (env[1:], with_untyped[1:])
    for aev in (radial_aev_numpy, _aev_jax):
        np.testing.assert_array_equal(aev(lig, env, with_untyped), aev(lig, *typed_only))
        assert np.all(aev(lig, env, np.array([0, 0])) == 0.0)


def test_isolated_ligand_atom_row_is_zero():
    lig = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])  # second atom is > 5.1 A from all
    env = np.array([[1.5, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 0.0, 3.5]])
    species = np.array([1, 12, 22])
    for aev in (radial_aev_numpy, _aev_jax):
        out = aev(lig, env, species)
        assert np.any(out[0] != 0.0)
        assert np.all(out[1] == 0.0)


def test_jax_matches_numpy(point_set):
    lig, env, species = point_set
    ref = radial_aev_numpy(lig, env, species)
    assert np.count_nonzero(ref.any(axis=1)) > 20  # most ligand points have neighbours

    nbr, mask = neighbour_lists(lig, env)
    out = jax.jit(radial_aev_jax)(lig, env, species, nbr, mask)
    assert out.shape == (30, 352)
    assert out.dtype == np.float32
    assert np.max(np.abs(np.asarray(out) - ref)) < 1e-5

    padded = _aev_jax(lig, env, species, k=nbr.shape[1] + 7)  # extra masked slots
    assert np.max(np.abs(padded - ref)) < 1e-5


def test_neighbour_lists_match_brute_force(point_set):
    lig, env, _ = point_set
    nbr, mask = neighbour_lists(lig, env)
    assert nbr.dtype == np.int32 and mask.dtype == np.bool_
    assert nbr.shape == mask.shape == (30, mask.sum(axis=1).max())
    dist = np.linalg.norm(lig[:, None, :] - env[None, :, :], axis=-1)
    for i in range(len(lig)):
        assert set(nbr[i, mask[i]].tolist()) == set(np.flatnonzero(dist[i] <= RC).tolist())


def test_neighbour_lists_raise_when_k_too_small(point_set):
    lig, env, _ = point_set
    kmax = int(neighbour_lists(lig, env)[1].sum(axis=1).max())
    nbr, mask = neighbour_lists(lig, env, k=kmax)
    assert nbr.shape == (30, kmax)
    with pytest.raises(ValueError):
        neighbour_lists(lig, env, k=kmax - 1)


def test_species_table_first_appearance_order():
    if not KEYS_CSV.exists():
        pytest.skip(f"{KEYS_CSV} not present (code-only mirror)")
    table = species_table(KEYS_CSV)
    assert len(table) == 187
    assert set(table.values()) == set(range(1, N_SPECIES + 1))
    expected = {"ALA-C": 1, "ALA-CA": 2, "ALA-CB": 3, "ALA-N": 4, "ALA-O": 5,
                "ALA-OXT": 5, "ARG-C": 1, "ARG-CB": 6, "ARG-CZ": 7, "TYR-OH": 22}
    assert {key: table[key] for key in expected} == expected
