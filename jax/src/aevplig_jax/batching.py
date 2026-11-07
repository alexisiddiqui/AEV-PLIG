"""Batch plans, static capacities and host-side batch assembly.

Plans follow PyTorch's scheme for the reproduction: one permutation of all ids per epoch from
``np.random.default_rng([seed, epoch])``, chunked by ``batch``, the last chunk short.

A ``Batch`` has static shapes set by ``Capacities``, so one compiled variant serves every batch:

- nodes: the ligand heavy atoms of all complexes concatenated (``ncap`` rows); the bond graph as
  padded neighbour lists (``kb`` slots), slot 0 the self-loop carrying the mean of the node's
  incoming bond features (PyG ``add_self_loops(fill_value="mean")``, zeros for an isolated node),
  slots 1.. the sources of its incoming bonds in edge order; unused slots point at the node itself
  with mask 0.
- environment: per complex the typed pocket atoms within ``RC`` of its ligand (``typed_within``),
  concatenated (``mcap`` rows); ``aev_nbr``/``aev_mask`` list, per ligand atom, the environment
  rows within ``RC`` (``ka`` slots) as indices into the concatenated arrays. Unused slots point at
  environment row 0 with mask 0.
- spread padding (jax-Ka): padded node rows copy real rows (row ``i`` copies row ``i mod n``, lists
  included) with ``node_valid = 0`` and ``node_graph = batch`` (the dump segment); padded
  environment rows copy real environment rows; padded batch slots have ``y = 0``, ``valid = 0``
  and no nodes.

Imports numpy, the standard library and package modules (scipy via ``aev.neighbour_lists``); no jax.
Arrays stay on the host: ``Prefetcher``'s ``to_device`` hook does the device transfer.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import NamedTuple

import numpy as np

from aevplig_jax.aev import neighbour_lists
from aevplig_jax.constants import BATCH, EDGE_DIM, RC, RDKIT_DIM
from aevplig_jax.pdb_atoms import typed_within
from aevplig_jax.store import Record, Store

NODE_QUANTUM = 256  # ncap is a multiple of this
ENV_QUANTUM = 1024  # mcap
KA_QUANTUM = 32  # ka


def _round_up(value: int, quantum: int) -> int:
    """The smallest multiple of ``quantum`` that is >= ``value`` and >= ``quantum`` (never zero)."""
    return max(1, -(-int(value) // quantum)) * quantum


@dataclass(frozen=True)
class Capacities:
    """Static batch shapes: ``ncap`` node rows, ``mcap`` environment rows, ``kb`` bond-list slots
    (largest in-degree + 1 self-loop), ``ka`` AEV-list slots, ``batch`` complexes."""

    ncap: int
    mcap: int
    kb: int
    ka: int
    batch: int = BATCH

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, value: dict) -> Capacities:
        return cls(**value)


class Batch(NamedTuple):
    """One assembled batch of numpy arrays with the shapes of its ``Capacities``.

    ``n_real`` (the number of real complexes) is host-only metadata, not an array: the trainer
    strips it before passing the batch to a jitted function.
    """

    x_lig: np.ndarray  # (ncap, RDKIT_DIM) float32
    node_graph: np.ndarray  # (ncap,) int32, batch slot of the node; padded rows -> batch
    node_valid: np.ndarray  # (ncap,) float32
    nbr: np.ndarray  # (ncap, kb) int32, slot 0 = self
    nbr_mask: np.ndarray  # (ncap, kb) float32
    nbr_feat: np.ndarray  # (ncap, kb, EDGE_DIM) float32
    lig_xyz: np.ndarray  # (ncap, 3) float32
    env_xyz: np.ndarray  # (mcap, 3) float32
    env_species: np.ndarray  # (mcap,) int32, 1..22
    aev_nbr: np.ndarray  # (ncap, ka) int32, rows of env_*
    aev_mask: np.ndarray  # (ncap, ka) float32
    y: np.ndarray  # (batch,) float32, standardised label; 0 on padded slots
    valid: np.ndarray  # (batch,) float32
    n_real: int


def _epoch_order(n: int, seed: int, epoch: int) -> np.ndarray:
    return np.random.default_rng([seed, epoch]).permutation(n)


def plan_epoch(ids: np.ndarray, seed: int, epoch: int, batch: int = BATCH) -> list[np.ndarray]:
    """``ids`` permuted by ``default_rng([seed, epoch])`` and cut into chunks of ``batch``."""
    ids = np.asarray(ids)
    order = _epoch_order(len(ids), seed, epoch)
    return [ids[order[start:start + batch]] for start in range(0, len(ids), batch)]


def _max_in_degree(store: Store) -> int:
    """The most bonds into one ligand atom over every record of ``store`` (one full read)."""
    most = 0
    for rid in store.ids:
        dst = store.get(rid).lig.bond_dst
        most = max(most, int(np.bincount(dst).max(initial=0)))
    return most


def capacities(store: Store, ids: np.ndarray, seeds: list[int], epochs: int,
               batch: int = BATCH) -> Capacities:
    """Capacities that fit every batch ``plan_epoch`` makes from ``ids`` for ``seeds`` x ``epochs``.

    ``ncap`` / ``mcap``: the largest per-batch total of ``L`` / ``A`` over those plans, rounded up
    to 256 / 1024. ``kb`` and ``ka`` cover every record of the store (so other splits of the same
    store fit too): ``kb`` = the largest in-degree + 1, from a scan of all records (the dims do not
    hold it); ``ka`` = the largest ``KA`` rounded up to 32.
    """
    ids = np.asarray(ids)
    if len(ids) == 0:
        raise ValueError("no ids to plan")
    row = {rid: i for i, rid in enumerate(store.ids)}
    sizes = store.dims[[row[str(rid)] for rid in ids]][:, [0, 3]].astype(np.int64)  # L, A
    starts = np.arange(0, len(ids), batch)
    most = np.zeros(2, dtype=np.int64)
    for seed in seeds:
        for epoch in range(epochs):
            totals = np.add.reduceat(sizes[_epoch_order(len(ids), seed, epoch)], starts, axis=0)
            most = np.maximum(most, totals.max(axis=0))
    return Capacities(ncap=_round_up(most[0], NODE_QUANTUM), mcap=_round_up(most[1], ENV_QUANTUM),
                      kb=_max_in_degree(store) + 1,
                      ka=_round_up(store.dims[:, 4].max(initial=0), KA_QUANTUM), batch=batch)


def _bond_lists(src: np.ndarray, dst: np.ndarray, bond_type: np.ndarray, ncap: int,
                kb: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``nbr``, ``nbr_mask``, ``nbr_feat`` of the bond graph (global node indices)."""
    nbr = np.tile(np.arange(ncap, dtype=np.int32)[:, None], (1, kb))
    nbr_mask = np.zeros((ncap, kb), np.float32)
    nbr_feat = np.zeros((ncap, kb, EDGE_DIM), np.float32)
    order = np.argsort(dst, kind="stable")  # group by destination, edge order within a group
    src, dst, bond_type = src[order], dst[order], bond_type[order]
    degree = np.bincount(dst, minlength=ncap)
    slot = np.arange(len(dst)) - (np.cumsum(degree) - degree)[dst] + 1
    nbr[dst, slot] = src
    nbr_mask[dst, slot] = 1.0
    nbr_feat[dst, slot] = bond_type
    incoming = np.zeros((ncap, EDGE_DIM), np.float32)
    np.add.at(incoming, dst, bond_type)
    nbr_mask[:, 0] = 1.0
    nbr_feat[:, 0] = incoming / np.maximum(degree, 1).astype(np.float32)[:, None]
    return nbr, nbr_mask, nbr_feat


def _spread(n_real: int, arrays: tuple[np.ndarray, ...]) -> None:
    """Spread padding in place: rows ``n_real..`` become copies of rows ``i mod n_real``."""
    rows = np.arange(n_real, len(arrays[0]))
    for array in arrays:
        array[rows] = array[rows % n_real]


def assemble(records: list[Record], cap: Capacities, y_mean: float, y_sd: float) -> Batch:
    """Pack ``records`` into one padded ``Batch``; ``y = (label - y_mean) / y_sd`` on real slots.

    Raises ``ValueError`` naming the complex when it would exceed a capacity (batch slots, ``ncap``,
    ``mcap``, a node in-degree over ``kb - 1``, a ligand atom with more than ``ka`` typed
    neighbours) or has no ligand atoms; nothing is truncated.
    """
    if not records:
        raise ValueError("no records to assemble")
    if len(records) > cap.batch:
        raise ValueError(f"{records[cap.batch].id}: {len(records)} complexes exceed batch "
                         f"{cap.batch}")
    x_lig = np.zeros((cap.ncap, RDKIT_DIM), np.float32)
    lig_xyz = np.zeros((cap.ncap, 3), np.float32)
    node_graph = np.full(cap.ncap, cap.batch, np.int32)
    node_valid = np.zeros(cap.ncap, np.float32)
    aev_nbr = np.zeros((cap.ncap, cap.ka), np.int32)
    aev_mask = np.zeros((cap.ncap, cap.ka), np.float32)
    env_xyz = np.zeros((cap.mcap, 3), np.float32)
    env_species = np.zeros(cap.mcap, np.int32)
    y = np.zeros(cap.batch, np.float32)
    valid = np.zeros(cap.batch, np.float32)
    bond_src, bond_dst, bond_type = [], [], []
    n_nodes = n_env = 0
    for slot, record in enumerate(records):
        lig, rid = record.lig, record.id
        n_lig = len(lig.xyz)
        if n_lig == 0:
            raise ValueError(f"{rid}: ligand has no atoms")
        if n_nodes + n_lig > cap.ncap:
            raise ValueError(f"{rid}: ligand atoms reach {n_nodes + n_lig}, over ncap {cap.ncap}")
        typed = typed_within(record.pocket, lig.xyz, cutoff=RC)
        n_typed = int(typed.sum())
        if n_env + n_typed > cap.mcap:
            raise ValueError(f"{rid}: environment atoms reach {n_env + n_typed}, over mcap "
                             f"{cap.mcap}")
        degree = int(np.bincount(lig.bond_dst).max(initial=0))
        if degree > cap.kb - 1:
            raise ValueError(f"{rid}: a ligand atom has {degree} incoming bonds, over kb - 1 = "
                             f"{cap.kb - 1}")
        nodes = slice(n_nodes, n_nodes + n_lig)
        if n_typed:
            try:
                nbr, mask = neighbour_lists(lig.xyz, record.pocket.xyz[typed], cutoff=RC, k=cap.ka)
            except ValueError as err:
                raise ValueError(f"{rid}: {err}") from err
            aev_nbr[nodes] = np.where(mask, nbr + n_env, 0)
            aev_mask[nodes] = mask
            env_xyz[n_env:n_env + n_typed] = record.pocket.xyz[typed]
            env_species[n_env:n_env + n_typed] = record.pocket.species[typed]
        x_lig[nodes] = lig.feat
        lig_xyz[nodes] = lig.xyz
        node_graph[nodes] = slot
        node_valid[nodes] = 1.0
        bond_src.append(lig.bond_src + n_nodes)
        bond_dst.append(lig.bond_dst + n_nodes)
        bond_type.append(lig.bond_type)
        y[slot] = (record.label - y_mean) / y_sd
        valid[slot] = 1.0
        n_nodes += n_lig
        n_env += n_typed

    nbr, nbr_mask, nbr_feat = _bond_lists(np.concatenate(bond_src), np.concatenate(bond_dst),
                                          np.concatenate(bond_type), cap.ncap, cap.kb)
    _spread(n_nodes, (x_lig, lig_xyz, nbr, nbr_mask, nbr_feat, aev_nbr, aev_mask))
    if n_env:  # with no environment atoms the rows stay zero (species 0 contributes nothing)
        _spread(n_env, (env_xyz, env_species))
    return Batch(x_lig=x_lig, node_graph=node_graph, node_valid=node_valid, nbr=nbr,
                 nbr_mask=nbr_mask, nbr_feat=nbr_feat, lig_xyz=lig_xyz, env_xyz=env_xyz,
                 env_species=env_species, aev_nbr=aev_nbr, aev_mask=aev_mask, y=y, valid=valid,
                 n_real=len(records))


class StoreSource:
    """``BatchSource`` over a ``Store``: ``load(spec)`` fetches the spec's records in a thread pool
    (``Store.get`` is thread-safe) and assembles them. ``close`` releases the pool; the store
    belongs to the caller.
    """

    def __init__(self, store: Store, cap: Capacities, y_mean: float, y_sd: float, workers: int):
        self.store, self.cap = store, cap
        self.y_mean, self.y_sd = float(y_mean), float(y_sd)
        self.workers = int(workers)
        self._pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="store")

    def load(self, spec: np.ndarray) -> Batch:
        records = list(self._pool.map(self.store.get, [str(rid) for rid in spec]))
        return assemble(records, self.cap, self.y_mean, self.y_sd)

    def close(self) -> None:
        self._pool.shutdown(wait=True)

    def provenance(self) -> dict:
        return {"store": str(self.store.path), "capacities": self.cap.to_json(),
                "y_mean": self.y_mean, "y_sd": self.y_sd, "workers": self.workers}
