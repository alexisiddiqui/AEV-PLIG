"""Paired-data layer: PairedDataset (.pt splits with row/series/lig/src ids), a series-aware
batch sampler and the within-series, different-ligand pair index."""
import os
import sys

sys.path.insert(0, os.environ["AEVPLIG_REPO"])

from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Sampler

try:
    from torch_geometric.data import Data, InMemoryDataset
except ImportError:  # the sampler and pair_index stay usable without PyG
    Data, InMemoryDataset = None, object

SOURCE_ID = {"pdbbind": 0, "bindingnet": 1, "bindingdb": 2}


def pair_index(series: torch.Tensor, lig: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """All (i, j) with i < j, series[i] == series[j] and lig[i] != lig[j], in lexicographic order."""
    n = series.numel()
    i, j = torch.triu_indices(n, n, offset=1, device=series.device)
    keep = (series[i] == series[j]) & (lig[i] != lig[j])
    return i[keep], j[keep]


class SeriesBatchSampler(Sampler):
    """Batches built from shuffled within-series chunks so series members share batches.

    Per epoch: permute each series, cut it into chunks of <= chunk, shuffle the chunks and pack
    them greedily into batches of <= batch_size (a chunk that does not fit starts the next batch).
    A trailing batch of one index is appended to the previous batch. The chunk is capped at
    batch_size - 1, which guarantees that no closed batch holds a single index.

    parent_aware (needs rows and parents, the row id and parent row id of every index): after the
    permutation each series is regrouped into families, a row with parent == row (an original row)
    followed by the indices whose parent is that row (its augmented rows), both in permuted order;
    indices whose parent is not an original row of the series follow as singletons. Whole families are
    packed greedily into chunks of <= chunk (a family that does not fit starts the next chunk; a family
    larger than chunk is cut into pieces of chunk, the first holding the original row), so an original
    row and up to chunk - 1 of its augmented rows always share a chunk. A series without augmented
    rows gets exactly the chunks of parent_aware=False. Every index once per epoch; deterministic per
    (seed, epoch).
    """

    def __init__(self, series: np.ndarray, batch_size: int = 128, chunk: int = 8, seed: int = 0,
                 parent_aware: bool = False, rows: np.ndarray | None = None,
                 parents: np.ndarray | None = None):
        if batch_size < 2:
            raise ValueError("batch_size must be >= 2 (BatchNorm needs two samples)")
        if chunk < 1:
            raise ValueError("chunk must be >= 1")
        self.series = np.asarray(series)
        self.batch_size = int(batch_size)
        self.chunk = min(int(chunk), self.batch_size - 1)
        self.seed = int(seed)
        self.epoch = 0
        self.parent_aware = bool(parent_aware)
        if self.parent_aware:
            if rows is None or parents is None:
                raise ValueError("parent_aware needs rows and parents")
            self.rows, self.parents = np.asarray(rows), np.asarray(parents)
            if self.rows.shape != self.series.shape or self.parents.shape != self.series.shape:
                raise ValueError("rows and parents must have one entry per index")
        order = np.argsort(self.series, kind="stable")
        _, starts = np.unique(self.series[order], return_index=True)
        self._groups = np.split(order, starts[1:]) if len(order) else []
        self._batches = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._batches = None

    def _build(self) -> list[list[int]]:
        rng = np.random.default_rng(self.seed * 1_000_003 + self.epoch)
        chunks = []
        for group in self._groups:
            perm = rng.permutation(group)
            if self.parent_aware:
                chunks.extend(self._family_chunks(perm))
            else:
                chunks.extend(perm[k:k + self.chunk] for k in range(0, len(perm), self.chunk))
        batches, current = [], []
        for k in rng.permutation(len(chunks)):
            piece = chunks[k]
            if current and len(current) + len(piece) > self.batch_size:
                batches.append(current)
                current = []
            current.extend(int(i) for i in piece)
        if current:
            batches.append(current)
        if len(batches) >= 2 and len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        return batches

    def _family_chunks(self, perm: np.ndarray) -> list[np.ndarray]:
        """Chunks of one permuted series: families (original row + its augmented rows) packed whole."""
        rows, parents = self.rows[perm], self.parents[perm]
        is_root = parents == rows
        roots = set(rows[is_root].tolist())
        children: dict = {}
        orphans = []
        for pos, par, root in zip(perm, parents, is_root):
            if root:
                continue
            if int(par) in roots:
                children.setdefault(int(par), []).append(pos)
            else:
                orphans.append(pos)
        families = [[pos] + children.get(int(r), []) for pos, r, root in zip(perm, rows, is_root) if root]
        families += [[pos] for pos in orphans]
        chunks, current = [], []
        for fam in families:
            if len(fam) > self.chunk:
                if current:
                    chunks.append(np.asarray(current))
                    current = []
                pieces = [fam[k:k + self.chunk] for k in range(0, len(fam), self.chunk)]
                chunks.extend(np.asarray(p) for p in pieces[:-1])
                current = list(pieces[-1])
                continue
            if len(current) + len(fam) > self.chunk:
                chunks.append(np.asarray(current))
                current = []
            current.extend(fam)
        if current:
            chunks.append(np.asarray(current))
        return chunks

    def _epoch_batches(self) -> list[list[int]]:
        if self._batches is None:
            self._batches = self._build()
        return self._batches

    def __iter__(self) -> Iterator[list[int]]:
        return iter([list(b) for b in self._epoch_batches()])

    def __len__(self) -> int:
        return len(self._epoch_batches())


def _graph_to_data(graph, pK: float, row: int, series: int, lig: int, src: int, kind: int = 0,
                   parent: int | None = None):
    """One AEV-PLIG pickle entry (c_size, features, edge_index, edge_features) -> Data.
    kind: 0 original row, 1 near-native pose, 2 decoy pose; parent: row of the original
    PDBbind row a pose was made from (row itself for kind 0)."""
    _, features, edge_index, edge_features = graph
    x = torch.as_tensor(np.asarray(features, dtype=np.float32))
    ei = np.asarray(edge_index, dtype=np.int64).reshape(-1, 2)
    ea = np.asarray(edge_features, dtype=np.float32)
    ea = ea.reshape(len(ei), -1) if ea.size else np.zeros((len(ei), 4), dtype=np.float32)
    return Data(
        x=x,
        edge_index=torch.as_tensor(ei).T.contiguous(),
        edge_attr=torch.as_tensor(ea),
        y=torch.tensor([pK], dtype=torch.float),
        row=torch.tensor([row], dtype=torch.long),
        series=torch.tensor([series], dtype=torch.long),
        lig=torch.tensor([lig], dtype=torch.long),
        src=torch.tensor([src], dtype=torch.long),
        kind=torch.tensor([kind], dtype=torch.long),
        parent=torch.tensor([row if parent is None else parent], dtype=torch.long),
    )


class PairedDataset(InMemoryDataset):
    """Loads root/processed/paired_<split>.pt. Never rebuilds: use PairedDataset.build to write it.
    Files written before arm 4 have no kind / parent fields; readers treat them as kind 0,
    parent = row."""

    def __init__(self, root: str, split: str, transform=None):
        self.split = split
        super().__init__(root, transform)
        path = self.processed_paths[0]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} missing; write it with PairedDataset.build")
        torch.serialization.add_safe_globals([Data])
        self.load(path)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f"paired_{self.split}.pt"]

    @classmethod
    def build(cls, root: str, split: str, rows, graphs: dict) -> "PairedDataset":
        """Write paired_<split>.pt from rows (columns unique_id, pK, row, series_id, lig_id,
        source, and optionally kind, parent: default 0 and row) in row order, taking graphs
        from graphs[unique_id]; return the loaded dataset."""
        kinds = rows["kind"] if "kind" in rows else [0] * len(rows)
        parents = rows["parent"] if "parent" in rows else rows["row"]
        data_list = [
            _graph_to_data(graphs[uid], float(pk), int(r), int(s), int(lg), SOURCE_ID[src],
                           int(kd), int(par))
            for uid, pk, r, s, lg, src, kd, par in zip(rows["unique_id"], rows["pK"], rows["row"],
                                                       rows["series_id"], rows["lig_id"],
                                                       rows["source"], kinds, parents)
        ]
        if not data_list:
            raise ValueError(f"split {split!r} has no rows")
        out_dir = os.path.join(root, "processed")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"paired_{split}.pt")
        tmp = path + ".tmp"
        cls.save(data_list, tmp)
        os.replace(tmp, path)
        del data_list
        return cls(root, split)
