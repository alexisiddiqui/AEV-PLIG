"""Radial AEV of ligand heavy atoms over protein neighbours (ANI-2x radial terms).

For ligand atom i, species s in 1..22 and shift k in 0..15::

    aev[i, (s-1)*16 + k] = sum_{j: species_j = s, r_ij < RC}
                           0.25 * exp(-ETA * (r_ij - SHIFTS[k])**2) * (0.5 * cos(pi * r_ij / RC) + 0.5)

The environment ``env_*`` is protein atoms only (the caller selects them); species 0 means
untyped and contributes nothing. This module imports numpy and scipy only; jax is imported
inside ``radial_aev_jax``.
"""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

from aevplig_jax.constants import AEV_DIM, ETA, N_SPECIES, RC, SHIFTS

if TYPE_CHECKING:
    import jax.numpy as jnp


def radial_aev_numpy(lig_xyz: np.ndarray, env_xyz: np.ndarray, env_species: np.ndarray) -> np.ndarray:
    """Brute-force float64 reference over all ligand-environment pairs, shape ``(L, AEV_DIM)``."""
    lig = np.asarray(lig_xyz, dtype=np.float64).reshape(-1, 3)
    env = np.asarray(env_xyz, dtype=np.float64).reshape(-1, 3)
    species = np.asarray(env_species).reshape(-1)
    r = np.linalg.norm(lig[:, None, :] - env[None, :, :], axis=-1)  # (L, P)
    fc = np.where(r < RC, 0.5 * np.cos(np.pi * r / RC) + 0.5, 0.0)
    terms = 0.25 * np.exp(-ETA * (r[..., None] - np.asarray(SHIFTS)) ** 2) * fc[..., None]  # (L, P, 16)
    aev = np.zeros((len(lig), N_SPECIES, len(SHIFTS)))
    for s in range(1, N_SPECIES + 1):
        aev[:, s - 1, :] = terms[:, species == s, :].sum(axis=1)
    return aev.reshape(len(lig), AEV_DIM)


def neighbour_lists(
    lig_xyz: np.ndarray, env_xyz: np.ndarray, cutoff: float = RC, k: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Environment atoms within ``cutoff`` (inclusive) of each ligand atom, from a cKDTree.

    Returns ``nbr (L, K) int32`` (sorted indices into ``env_xyz``, padded with 0) and
    ``mask (L, K) bool`` (True on real slots). ``K`` is the largest count when ``k`` is None,
    else ``k``; raises ``ValueError`` when any ligand atom has more than ``k`` neighbours.
    """
    lig = np.asarray(lig_xyz, dtype=np.float64).reshape(-1, 3)
    env = np.asarray(env_xyz, dtype=np.float64).reshape(-1, 3)
    rows = cKDTree(env).query_ball_point(lig, r=cutoff, return_sorted=True)
    counts = np.fromiter((len(row) for row in rows), dtype=np.int64, count=len(lig))
    most = int(counts.max(initial=0))
    width = most if k is None else k
    if most > width:
        raise ValueError(
            f"{int((counts > width).sum())} ligand atoms have more than k={width} neighbours "
            f"within {cutoff} A (largest count {most})"
        )
    mask = np.arange(width) < counts[:, None]
    nbr = np.zeros((len(lig), width), dtype=np.int32)
    nbr[mask] = np.fromiter(chain.from_iterable(rows), dtype=np.int32, count=int(counts.sum()))
    return nbr, mask


def radial_aev_jax(lig_xyz, env_xyz, env_species, nbr, mask) -> jnp.ndarray:
    """Radial AEV in float32 from padded neighbour lists, shape ``(L, AEV_DIM)``.

    Gathers the neighbours by ``nbr``, zeroes slots where ``mask`` is False or ``r >= RC``,
    and sums the terms into species channels with a one-hot over species (0 maps to no channel).
    """
    import jax
    import jax.numpy as jnp

    lig = jnp.asarray(lig_xyz, dtype=jnp.float32)
    env = jnp.asarray(env_xyz, dtype=jnp.float32)[nbr]  # (L, K, 3)
    species = jnp.asarray(env_species, dtype=jnp.int32)[nbr]  # (L, K)
    r = jnp.sqrt(jnp.sum((env - lig[:, None, :]) ** 2, axis=-1))  # (L, K)
    keep = jnp.asarray(mask) & (r < RC)
    fc = jnp.where(keep, 0.5 * jnp.cos(r * (jnp.pi / RC)) + 0.5, 0.0)
    shifts = jnp.asarray(SHIFTS, dtype=jnp.float32)
    terms = 0.25 * jnp.exp(-ETA * (r[..., None] - shifts) ** 2) * fc[..., None]  # (L, K, 16)
    onehot = jax.nn.one_hot(species - 1, N_SPECIES, dtype=jnp.float32)  # (L, K, 22)
    # Full float32 contraction: the default GPU matmul precision (TF32) would break AEV parity.
    aev = jnp.einsum("lks,lkr->lsr", onehot, terms, precision=jax.lax.Precision.HIGHEST)
    return aev.reshape(lig.shape[0], AEV_DIM)
