"""Shared by train, predict and precompute_aev: checkpoint files, sequential prediction batches
and the capacities that fit them, evaluation-mode prediction, de-standardisation and helpers.py's
Pearson and RMSE.

A checkpoint is one ``state.npz``: the leaves of named pytrees under ``<name><keystr(path)>`` keys
(e.g. ``params['gat'][0]['w_l']``) plus a JSON ``__meta__`` entry, written to a temporary file and
renamed into place. Loading fills templates (``jax.eval_shape`` results or real trees) leaf by
leaf, so no pickle is involved.

Prediction batches cut the ids in order into chunks of ``BATCH`` (the original's
``shuffle=False`` loaders). ``plan_capacities`` sizes them from the prepared records being
predicted, never from a training manifest, so any ligand fits.

Imports jax, numpy, the standard library and package modules.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import jax
import numpy as np

from aevplig_jax import model
from aevplig_jax.batching import (ENV_QUANTUM, KA_QUANTUM, NODE_QUANTUM, Batch, Capacities,
                                  Prepared, _round_up)
from aevplig_jax.constants import BATCH
from aevplig_jax.loading import LoaderConfig, Prefetcher

META_KEY = "__meta__"


def enable_compilation_cache() -> None:
    """Use the persistent compilation cache in ``$JAX_COMPILATION_CACHE_DIR`` when it is set."""
    cache = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if cache:
        jax.config.update("jax_compilation_cache_dir", cache)


def save_state(path: Path, trees: dict[str, Any], meta: dict) -> None:
    """Write ``trees`` (name -> pytree) and the JSON-ready ``meta`` to ``path`` atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, tree in trees.items():
        leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
        arrays.update({name + jax.tree_util.keystr(key): np.asarray(leaf) for key, leaf in leaves})
    arrays[META_KEY] = np.array(json.dumps(meta))
    pending = path.with_name(path.name + ".tmp")
    with open(pending, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(pending, path)


def load_state(path: Path, templates: dict[str, Any]) -> tuple[dict[str, Any], dict]:
    """The trees of ``path`` named in ``templates``, shaped like them with device arrays as leaves,
    and the meta dict. Other arrays in the file are ignored; a missing leaf raises ``KeyError``, a
    shape or dtype unlike the template's ``ValueError``."""
    trees = {}
    with np.load(path) as data:
        meta = json.loads(str(data[META_KEY]))
        for name, template in templates.items():
            leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
            values = []
            for key, leaf in leaves:
                field = name + jax.tree_util.keystr(key)
                if field not in data.files:
                    raise KeyError(f"{path}: no array {field}")
                value = data[field]
                if value.shape != tuple(leaf.shape) or value.dtype != np.dtype(leaf.dtype):
                    raise ValueError(f"{path}: {field} is {value.dtype} {value.shape}, expected "
                                     f"{np.dtype(leaf.dtype)} {tuple(leaf.shape)}")
                values.append(jax.device_put(value))
            trees[name] = jax.tree_util.tree_unflatten(treedef, values)
    return trees, meta


def member_templates() -> dict[str, Any]:
    """Shapes of the ``params`` and ``bn_state`` trees a ``best`` checkpoint holds."""
    return {"params": jax.eval_shape(model.init_params, jax.random.key(0)),
            "bn_state": jax.eval_shape(model.init_bn_state)}


def load_member(path: Path) -> tuple[dict, dict, dict]:
    """``(params, bn_state, meta)`` of a ``best`` checkpoint file."""
    trees, meta = load_state(path, member_templates())
    return trees["params"], trees["bn_state"], meta


def sequential_plans(ids, batch: int = BATCH) -> list[np.ndarray]:
    """``ids`` in order, cut into chunks of ``batch`` (the last one short)."""
    ids = np.asarray(ids)
    return [ids[start:start + batch] for start in range(0, len(ids), batch)]


def plan_capacities(prepared: dict[str, Prepared], plans, batch: int = BATCH) -> Capacities:
    """The smallest capacities (batching's quanta) that fit every plan: ``ncap`` / ``mcap`` from
    the largest per-plan totals of ligand and environment atoms, ``kb`` the widest bond list
    (largest in-degree + 1), ``ka`` the widest AEV list."""
    ncap = mcap = ka = 0
    kb = 1
    for plan in plans:
        items = [prepared[str(rid)] for rid in plan]
        ncap = max(ncap, sum(len(item.xyz) for item in items))
        mcap = max(mcap, sum(len(item.env_xyz) for item in items))
        kb = max([kb, *(item.nbr.shape[1] for item in items)])
        ka = max([ka, *(item.aev_nbr.shape[1] for item in items)])
    return Capacities(ncap=_round_up(ncap, NODE_QUANTUM), mcap=_round_up(mcap, ENV_QUANTUM), kb=kb,
                      ka=_round_up(ka, KA_QUANTUM), batch=batch)


def merge_capacities(*caps: Capacities) -> Capacities:
    """Field-wise maximum of capacities with the same ``batch``."""
    if len({cap.batch for cap in caps}) != 1:
        raise ValueError(f"capacities for different batch sizes: {caps}")
    return Capacities(ncap=max(c.ncap for c in caps), mcap=max(c.mcap for c in caps),
                      kb=max(c.kb for c in caps), ka=max(c.ka for c in caps), batch=caps[0].batch)


def device_batch(batch: Batch) -> tuple[Batch, int]:
    """``Prefetcher``'s ``to_device``: the array fields on the device, ``n_real`` set to None (an
    empty pytree node, so it never reaches jit), and the host ``n_real``."""
    return jax.device_put(batch._replace(n_real=None)), batch.n_real


def predict_function(features: Callable = model.aev_features) -> Callable:
    """The jitted ``predict_fn(params, bn_state, batch) -> pred`` (evaluation mode, standardised)."""

    @jax.jit
    def predict_fn(params, bn_state, batch):
        return model.predict(params, bn_state, batch, False, features)[0]

    return predict_fn


def predict_ids(predict_fn: Callable, members: list[tuple[dict, dict]], source, plans,
                config: LoaderConfig) -> np.ndarray:
    """Standardised predictions ``(len(members), n_ids)`` of each ``(params, bn_state)`` member
    over ``plans`` in order; each batch is assembled and moved to the device once for all members."""
    counts, outputs = [], [[] for _ in members]
    for _, (batch, n_real) in Prefetcher(source, plans, config, to_device=device_batch):
        counts.append(n_real)
        for out, (params, bn_state) in zip(outputs, members):
            out.append(predict_fn(params, bn_state, batch))
    host = jax.device_get(outputs)
    return np.array([np.concatenate([pred[:n] for pred, n in zip(rows, counts)]) if counts
                     else np.zeros(0) for rows in host], dtype=np.float64)


def destandardise(pred, y_mean: float, y_sd: float) -> np.ndarray:
    """``StandardScaler.inverse_transform``: ``pred * y_sd + y_mean`` in float64."""
    return np.asarray(pred, dtype=np.float64) * y_sd + y_mean


def pearson(y, f) -> float:
    """helpers.pearson, ``np.corrcoef(y, f)[0, 1]`` (NaN for constant input, without a warning)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.corrcoef(y, f)[0, 1])


def rmse(y, f) -> float:
    """helpers.rmse, ``sqrt(((y - f) ** 2).mean(axis=0))``."""
    y, f = np.asarray(y, dtype=np.float64), np.asarray(f, dtype=np.float64)
    return math.sqrt(((y - f) ** 2).mean(axis=0))
