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
  rows within ``RC`` (``ka`` slots) as indices into the concatenated arrays, in the form
  ``aev.neighbour_lists`` gives (ascending, padded). Unused slots point at environment row 0 with
  mask 0.
- spread padding (jax-Ka): padded node rows copy real rows (row ``i`` copies row ``i mod n``, lists
  included) with ``node_valid = 0`` and ``node_graph = batch`` (the dump segment); padded
  environment rows copy real environment rows; padded batch slots have ``y = 0``, ``valid = 0``
  and no nodes.

Work is split in two. ``prepare`` turns one record into compact local arrays (``Prepared``: the
typed subset and AEV lists from one ``typed_pairs`` pass, bond lists and self-loop features);
``prepare_all`` runs it once per record before training, in a fork pool of processes (threads would
serialise on the GIL). Per batch, ``_pad`` only checks capacities, offsets, concatenates and pads;
``StoreSource.load`` (from the prepared records) and ``assemble`` (raw records, prepared inline)
both go through it.

Imports numpy, the standard library and package modules; no jax. Arrays stay on the host:
``Prefetcher``'s ``to_device`` hook does the device transfer.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from aevplig_jax.constants import BATCH, EDGE_DIM, RC, RDKIT_DIM
from aevplig_jax.pdb_atoms import typed_pairs
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


@dataclass(frozen=True, eq=False)
class Prepared:
    """One complex, prepared once for every batch it joins, as compact local arrays.

    Bond lists: ``nbr`` has ``1 + max(degree)`` slots; slot 0 is the atom itself, slots
    ``1..degree`` the sources of its incoming bonds in edge order, the rest the atom itself.
    ``nbr_feat`` slot 0 is the mean of the incoming bond features (zeros when isolated). AEV lists:
    row ``i`` of ``aev_nbr`` holds ``aev_count[i]`` ascending indices into ``env_*``, then zeros.
    ``nbr`` and ``aev_nbr`` are int16 when their indices fit, int32 otherwise.
    """

    id: str
    label: float
    feat: np.ndarray  # (L, RDKIT_DIM) float32
    xyz: np.ndarray  # (L, 3) float32
    degree: np.ndarray  # (L,) int32, incoming bonds
    nbr: np.ndarray  # (L, Wb) int16
    nbr_feat: np.ndarray  # (L, Wb, EDGE_DIM) float32
    env_xyz: np.ndarray  # (A, 3) float32, the typed_within subset in pocket order
    env_species: np.ndarray  # (A,) int8
    aev_count: np.ndarray  # (L,) int32, typed neighbours within RC
    aev_nbr: np.ndarray  # (L, Wa) int16


def _index_dtype(size: int) -> type:
    """int16 when indices ``0..size-1`` fit, else int32."""
    return np.int16 if size <= np.iinfo(np.int16).max + 1 else np.int32


def _packed(groups: np.ndarray, counts: np.ndarray, width: int, first: int = 0) -> np.ndarray:
    """Flat positions in a C-ordered ``(len(counts), width)`` array that pack each group's entries
    to the left from slot ``first`` (``groups`` sorted ascending, ``counts`` per group)."""
    return groups * width + np.arange(len(groups)) - (np.cumsum(counts) - counts)[groups] + first


def prepare(record: Record) -> Prepared:
    """The ``Prepared`` arrays of one record: bond neighbour lists with the self-loop feature, and
    the typed environment with its AEV lists from one ``typed_pairs`` pass.
    """
    lig = record.lig
    n_lig = len(lig.xyz)
    degree = np.bincount(lig.bond_dst, minlength=n_lig)
    width = 1 + int(degree.max(initial=0))
    nbr = np.repeat(np.arange(n_lig, dtype=_index_dtype(n_lig))[:, None], width, axis=1)
    nbr_feat = np.zeros((n_lig, width, EDGE_DIM), np.float32)
    if len(lig.bond_dst):
        order = np.argsort(lig.bond_dst, kind="stable")  # by destination, edge order within
        dst = lig.bond_dst[order]
        flat = _packed(dst, degree, width, first=1)
        nbr.ravel()[flat] = lig.bond_src[order]
        nbr_feat.reshape(-1, EDGE_DIM)[flat] = lig.bond_type[order]
        incoming = nbr_feat[:, 1:].sum(axis=1)  # exact: sums of one-hot rows
        nbr_feat[:, 0] = incoming / np.maximum(degree, 1).astype(np.float32)[:, None]

    lig_index, pocket_index = typed_pairs(record.pocket, lig.xyz, cutoff=RC)
    typed = np.zeros(len(record.pocket.xyz), dtype=bool)
    typed[pocket_index] = True
    env_rows = np.flatnonzero(typed)
    env_rank = np.cumsum(typed, dtype=np.int32) - 1  # pocket row -> environment row
    counts = np.bincount(lig_index, minlength=n_lig)
    aev_nbr = np.zeros((n_lig, int(counts.max(initial=0))), _index_dtype(len(env_rows)))
    aev_nbr.ravel()[_packed(lig_index, counts, aev_nbr.shape[1])] = env_rank[pocket_index]
    return Prepared(id=record.id, label=float(record.label), feat=lig.feat, xyz=lig.xyz,
                    degree=degree.astype(np.int32), nbr=nbr, nbr_feat=nbr_feat,
                    env_xyz=record.pocket.xyz[env_rows],
                    env_species=record.pocket.species[env_rows],
                    aev_count=counts.astype(np.int32), aev_nbr=aev_nbr)


def _fork_context():
    """The ``fork`` multiprocessing context, or None where the platform has none."""
    try:
        return multiprocessing.get_context("fork")
    except ValueError:
        return None


_worker_store: Store | None = None  # a pool worker's own Store, opened after the fork


def _open_worker_store(path: str) -> None:
    global _worker_store
    _worker_store = Store(Path(path))


def _prepare_chunk(ids: list[str]) -> list[Prepared]:
    return [prepare(_worker_store.get(rid)) for rid in ids]


def prepare_all(store: Store, ids, workers: int = 1) -> dict[str, Prepared]:
    """``prepare`` every id once; the dict follows ``ids`` order and does not depend on ``workers``.

    With ``workers > 1`` and the fork start method available, chunks of ids are prepared by a
    ``ProcessPoolExecutor`` of ``workers`` forked processes, each opening its own ``Store`` on
    ``store.path`` (no descriptor or decompressor is shared with the parent); otherwise inline. A
    worker that fails to start or dies raises ``BrokenProcessPool`` here instead of hanging. Call it
    before jax starts its threads: forking a process that runs jax warns and can deadlock.
    """
    ids = [str(rid) for rid in ids]
    context = _fork_context() if workers > 1 else None
    if context is None:
        return {rid: prepare(store.get(rid)) for rid in ids}
    size = max(1, -(-len(ids) // (4 * workers)))  # about four chunks per worker
    chunks = [ids[start:start + size] for start in range(0, len(ids), size)]
    prepared: dict[str, Prepared] = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                             initializer=_open_worker_store,
                             initargs=(str(store.path),)) as executor:
        for chunk, items in zip(chunks, executor.map(_prepare_chunk, chunks)):  # in chunk order
            prepared.update(zip(chunk, items))
    return prepared


def _spread(n_real: int, arrays: tuple[np.ndarray, ...]) -> None:
    """Spread padding in place: rows ``n_real..`` become copies of rows ``i mod n_real``."""
    rows = np.arange(n_real, len(arrays[0]))
    for array in arrays:
        array[rows] = array[rows % n_real]


def _pad(items: list[Prepared], cap: Capacities, y_mean: float, y_sd: float) -> Batch:
    """The batch layout, shared by ``assemble`` and ``StoreSource.load``: checks capacities, offsets
    and concatenates the prepared complexes, and spread-pads to ``cap``."""
    if not items:
        raise ValueError("no records to assemble")
    if len(items) > cap.batch:
        raise ValueError(f"{items[cap.batch].id}: {len(items)} complexes exceed batch {cap.batch}")
    nbr = np.repeat(np.arange(cap.ncap, dtype=np.int32)[:, None], cap.kb, axis=1)
    nbr_mask = np.zeros((cap.ncap, cap.kb), np.float32)
    nbr_feat = np.zeros((cap.ncap, cap.kb, EDGE_DIM), np.float32)
    aev_nbr = np.zeros((cap.ncap, cap.ka), np.int32)
    aev_mask = np.zeros((cap.ncap, cap.ka), np.float32)
    sizes = np.zeros(len(items), np.int64)
    n_nodes = n_env = 0
    for slot, item in enumerate(items):  # checks and the variable-width lists
        n_lig, n_typed = len(item.xyz), len(item.env_xyz)
        width_b, width_a = item.nbr.shape[1], item.aev_nbr.shape[1]
        if n_lig == 0:
            raise ValueError(f"{item.id}: ligand has no atoms")
        if n_nodes + n_lig > cap.ncap:
            raise ValueError(f"{item.id}: ligand atoms reach {n_nodes + n_lig}, over ncap "
                             f"{cap.ncap}")
        if n_env + n_typed > cap.mcap:
            raise ValueError(f"{item.id}: environment atoms reach {n_env + n_typed}, over mcap "
                             f"{cap.mcap}")
        if width_b > cap.kb:
            raise ValueError(f"{item.id}: a ligand atom has {width_b - 1} incoming bonds, over "
                             f"kb - 1 = {cap.kb - 1}")
        if width_a > cap.ka:
            raise ValueError(f"{item.id}: a ligand atom has {width_a} typed neighbours within "
                             f"{RC} A, over ka = {cap.ka}")
        rows = slice(n_nodes, n_nodes + n_lig)
        nbr[rows, :width_b] = item.nbr + np.int32(n_nodes)
        nbr_mask[rows, :width_b] = np.arange(width_b) <= item.degree[:, None]
        nbr_feat[rows, :width_b] = item.nbr_feat
        if width_a:
            filled = np.arange(width_a) < item.aev_count[:, None]
            aev_nbr[rows, :width_a] = np.where(filled, item.aev_nbr + np.int32(n_env), 0)
            aev_mask[rows, :width_a] = filled
        sizes[slot] = n_lig
        n_nodes += n_lig
        n_env += n_typed

    x_lig = np.zeros((cap.ncap, RDKIT_DIM), np.float32)
    lig_xyz = np.zeros((cap.ncap, 3), np.float32)
    x_lig[:n_nodes] = np.concatenate([item.feat for item in items])
    lig_xyz[:n_nodes] = np.concatenate([item.xyz for item in items])
    node_graph = np.full(cap.ncap, cap.batch, np.int32)
    node_graph[:n_nodes] = np.repeat(np.arange(len(items), dtype=np.int32), sizes)
    node_valid = np.zeros(cap.ncap, np.float32)
    node_valid[:n_nodes] = 1.0
    env_xyz = np.zeros((cap.mcap, 3), np.float32)
    env_species = np.zeros(cap.mcap, np.int32)
    if n_env:
        env_xyz[:n_env] = np.concatenate([item.env_xyz for item in items])
        env_species[:n_env] = np.concatenate([item.env_species for item in items])
    y = np.zeros(cap.batch, np.float32)
    y[:len(items)] = (np.array([item.label for item in items]) - y_mean) / y_sd
    valid = np.zeros(cap.batch, np.float32)
    valid[:len(items)] = 1.0

    _spread(n_nodes, (x_lig, lig_xyz, nbr, nbr_mask, nbr_feat, aev_nbr, aev_mask))
    if n_env:  # with no environment atoms the rows stay zero (species 0 contributes nothing)
        _spread(n_env, (env_xyz, env_species))
    return Batch(x_lig=x_lig, node_graph=node_graph, node_valid=node_valid, nbr=nbr,
                 nbr_mask=nbr_mask, nbr_feat=nbr_feat, lig_xyz=lig_xyz, env_xyz=env_xyz,
                 env_species=env_species, aev_nbr=aev_nbr, aev_mask=aev_mask, y=y, valid=valid,
                 n_real=len(items))


def assemble(records: list[Record | Prepared], cap: Capacities, y_mean: float,
             y_sd: float) -> Batch:
    """Pack complexes into one padded ``Batch``; ``y = (label - y_mean) / y_sd`` on real slots.

    Raw records are prepared inline (``Prepared`` items pass through). Raises ``ValueError`` naming
    the complex when it would exceed a capacity (batch slots, ``ncap``, ``mcap``, an in-degree over
    ``kb - 1``, a ligand atom with more than ``ka`` typed neighbours) or has no ligand atoms;
    nothing is truncated.
    """
    return _pad([r if isinstance(r, Prepared) else prepare(r) for r in records], cap, y_mean, y_sd)


class StoreSource:
    """``BatchSource`` over prepared records: ``load(spec)`` gathers the spec's ``Prepared`` records
    and pads them, with no store reads and no per-record work at batch time.

    ``prepared`` is a ``prepare_all`` result; when it is None the constructor builds it for ``ids``
    (default: every store id) with ``workers`` processes. ``close`` drops the reference to it; the
    store belongs to the caller.
    """

    def __init__(self, store: Store, cap: Capacities, y_mean: float, y_sd: float, workers: int,
                 prepared: dict[str, Prepared] | None = None, ids=None):
        self.store, self.cap = store, cap
        self.y_mean, self.y_sd = float(y_mean), float(y_sd)
        self.workers = int(workers)
        if prepared is None:
            prepared = prepare_all(store, store.ids if ids is None else ids, self.workers)
        self.prepared = prepared

    def load(self, spec: np.ndarray) -> Batch:
        return _pad([self.prepared[str(rid)] for rid in spec], self.cap, self.y_mean, self.y_sd)

    def close(self) -> None:
        self.prepared = {}

    def provenance(self) -> dict:
        return {"store": str(self.store.path), "capacities": self.cap.to_json(),
                "y_mean": self.y_mean, "y_sd": self.y_sd, "workers": self.workers,
                "prepared": len(self.prepared)}
