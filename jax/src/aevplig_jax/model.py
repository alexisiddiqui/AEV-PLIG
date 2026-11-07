"""AEV-PLIG's GATv2Net (model_defs.py) in JAX on the padded batches of batching.py.

Parameters are nested dicts of float32 arrays:

- ``params["gat"][i]``: ``w_l``, ``b_l``, ``w_r``, ``b_r`` (``(in, heads*hidden)`` and
  ``(heads*hidden,)``), ``w_e`` ``(edge_dim, heads*hidden)``, ``att`` ``(heads, hidden)``, ``bias``
  ``(heads*hidden,)`` and the layer's BatchNorm ``bn = {"scale", "shift"}``;
- ``params["fc1"|"fc2"|"fc3"|"out"] = {"w": (in, out), "b": (out,)}`` and
  ``params["bn1"|"bn2"|"bn3"] = {"scale", "shift"}``;
- ``params["heads"]``: reserved for later heads (``init_head``), empty here.

Keys: the trunk draws ``split(key, layers + 4)``, which under partitionable threefry equals
``fold_in(key, i)`` for ``i < layers + 4``. Every later branch therefore draws inside its own
namespace, ``fold_in(fold_in(key, NS), salt)`` with a module constant ``NS`` at or above the
trunk's split count: heads use ``HEADS_NS`` (``head_key``); a later branch such as a pocket
encoder takes its own constant (e.g. ``POCKET_NS = HEADS_NS + 1``) the same way. The trunk's own
draws are unchanged, so adding a branch never changes the trunk's initialisation.

BatchNorm running statistics are a separate tree, ``state`` (``init_bn_state``), returned updated by
every function that normalises in training mode. Initialisation matches the original at the
distribution level: ``utils.init_weights`` puts Xavier normal on every ``weight`` outside BatchNorm,
the rest keeps the PyG 2.5.3 / torch defaults (glorot uniform ``att``, zero conv bias,
``U(-1/sqrt(fan_in), 1/sqrt(fan_in))`` linear biases).

Imports jax and package modules only (constants, aev).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from aevplig_jax.aev import radial_aev_jax
from aevplig_jax.constants import (ACT_SLOPE, ATTN_SLOPE, BN_EPS, BN_MOMENTUM, DENSE, EDGE_DIM,
                                   HEADS, HIDDEN, LAYERS, MASK_FILL, NODE_DIM)

DENSE_NAMES = ("fc1", "fc2", "fc3", "out")  # widths DENSE, from 2 * heads * hidden
NORM_NAMES = ("bn1", "bn2", "bn3")  # after fc1, fc2, fc3
HEADS_NS = 1000  # key namespace of init_head; above the trunk's split count (LAYERS + 4 = 9)


def _xavier_normal(key, fan_in: int, fan_out: int) -> jnp.ndarray:
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return std * jax.random.normal(key, (fan_in, fan_out), jnp.float32)


def _fan_in_uniform(key, fan_in: int, size: int) -> jnp.ndarray:
    bound = 1.0 / math.sqrt(fan_in)
    return jax.random.uniform(key, (size,), jnp.float32, -bound, bound)


def _dense(key, fan_in: int, fan_out: int) -> dict:
    """``{"w", "b"}``: Xavier normal weight, ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))`` bias."""
    key_w, key_b = jax.random.split(key)
    return {"w": _xavier_normal(key_w, fan_in, fan_out),
            "b": _fan_in_uniform(key_b, fan_in, fan_out)}


def _norm(width: int) -> dict:
    return {"scale": jnp.ones(width, jnp.float32), "shift": jnp.zeros(width, jnp.float32)}


def _running(width: int) -> dict:
    return {"mean": jnp.zeros(width, jnp.float32), "var": jnp.ones(width, jnp.float32)}


def _gat(key, fan_in: int, edge_dim: int, hidden: int, heads: int) -> dict:
    width = heads * hidden
    key_l, key_r, key_e, key_att = jax.random.split(key, 4)
    left, right = _dense(key_l, fan_in, width), _dense(key_r, fan_in, width)
    bound = math.sqrt(6.0 / (heads + hidden))  # PyG glorot on att (1, heads, hidden)
    return {"w_l": left["w"], "b_l": left["b"], "w_r": right["w"], "b_r": right["b"],
            "w_e": _xavier_normal(key_e, edge_dim, width),
            "att": jax.random.uniform(key_att, (heads, hidden), jnp.float32, -bound, bound),
            "bias": jnp.zeros(width, jnp.float32), "bn": _norm(width)}


def init_params(key, node_dim: int = NODE_DIM, edge_dim: int = EDGE_DIM, hidden: int = HIDDEN,
                heads: int = HEADS, layers: int = LAYERS) -> dict:
    """The parameter tree; each GATv2 layer and each dense layer draws from its own split key."""
    if layers + len(DENSE_NAMES) > HEADS_NS:
        raise ValueError(f"{layers} layers: the trunk's keys would reach the heads' namespace")
    keys = jax.random.split(key, layers + len(DENSE_NAMES))
    width = heads * hidden
    params = {"gat": [_gat(keys[i], node_dim if i == 0 else width, edge_dim, hidden, heads)
                      for i in range(layers)],
              "heads": {}}
    sizes = (2 * width, *DENSE)
    for j, name in enumerate(DENSE_NAMES):
        params[name] = _dense(keys[layers + j], sizes[j], sizes[j + 1])
    for j, name in enumerate(NORM_NAMES):
        params[name] = _norm(DENSE[j])
    return params


def init_bn_state(layers: int = LAYERS) -> dict:
    """Running mean 0 and variance 1 for the ``layers`` node norms and the three dense norms."""
    state = {"gat": [_running(HEADS * HIDDEN) for _ in range(layers)]}
    for j, name in enumerate(NORM_NAMES):
        state[name] = _running(DENSE[j])
    return state


def head_key(key, salt: int):
    """The key of the head with ``salt``: ``fold_in(fold_in(key, HEADS_NS), salt)``, never one of
    the trunk's keys (a plain ``fold_in(key, salt)`` equals the trunk's ``split`` key ``salt``)."""
    return jax.random.fold_in(jax.random.fold_in(key, HEADS_NS), salt)


def init_head(key, name: str, salt: int, shape: tuple[int, int]) -> dict:
    """``{name: {"w": shape, "b": (shape[1],)}}`` to merge into ``params["heads"]``, initialised
    like the dense layers from ``head_key(key, salt)``, so adding a head never changes the trunk
    and never copies one of its blocks."""
    fan_in, fan_out = shape
    return {name: _dense(head_key(key, salt), fan_in, fan_out)}


def gatv2_layer(p: dict, x, nbr, nbr_mask, nbr_feat, heads: int, hidden: int) -> jnp.ndarray:
    """PyG ``GATv2Conv`` (``share_weights=False``, ``add_self_loops`` with ``fill_value="mean"``,
    no dropout, concatenated heads) in neighbour-list form, shape ``(ncap, heads*hidden)``.

    Target ``i`` attends over its sources ``nbr[i, k]`` (slot 0 the self-loop) where
    ``nbr_mask > 0``: ``e_ik = att . leaky_relu_0.2(hl[nbr[i, k]] + hr[i] + he[i, k])``, softmax
    over ``k``, message ``sum_k alpha_ik * hl[nbr[i, k]]``, plus ``bias``.
    """
    n, k = nbr.shape
    hl = (x @ p["w_l"] + p["b_l"]).reshape(n, heads, hidden)
    hr = (x @ p["w_r"] + p["b_r"]).reshape(n, heads, hidden)
    he = (nbr_feat @ p["w_e"]).reshape(n, k, heads, hidden)
    source = hl[nbr]  # (n, k, heads, hidden)
    e = jax.nn.leaky_relu(source + hr[:, None] + he, ATTN_SLOPE)
    logits = jnp.where(nbr_mask[..., None] > 0, (e * p["att"]).sum(-1), MASK_FILL)  # (n, k, heads)
    alpha = jax.nn.softmax(logits, axis=1)
    out = (alpha[..., None] * source).sum(axis=1)
    return out.reshape(n, heads * hidden) + p["bias"]


def masked_batchnorm(p: dict, state: dict, x, valid, train: bool, momentum: float = BN_MOMENTUM,
                     eps: float = BN_EPS) -> tuple[jnp.ndarray, dict]:
    """``torch.nn.BatchNorm1d`` over the rows of ``x`` with ``valid > 0``.

    Training: normalise with the batch mean and biased variance of the valid rows, and move the
    running mean and the running unbiased variance by ``momentum``. Evaluation: normalise with the
    running statistics and return ``state`` unchanged. Every row is normalised; invalid rows are
    only left out of the statistics.
    """
    if train:
        keep = (valid > 0)[:, None]
        count = jnp.sum(keep).astype(jnp.float32)
        n = jnp.maximum(count, 1.0)
        mean = jnp.where(keep, x, 0.0).sum(axis=0) / n
        var = jnp.where(keep, (x - mean) ** 2, 0.0).sum(axis=0) / n
        unbiased = var * n / jnp.maximum(n - 1.0, 1.0)
        state = {"mean": (1.0 - momentum) * state["mean"] + momentum * mean,
                 "var": (1.0 - momentum) * state["var"] + momentum * unbiased}
    else:
        mean, var = state["mean"], state["var"]
    return (x - mean) * jax.lax.rsqrt(var + eps) * p["scale"] + p["shift"], state


def aev_features(batch) -> jnp.ndarray:
    """The 367 node features: ``x_lig`` then the radial AEV over the batch's environment lists."""
    aev = radial_aev_jax(batch.lig_xyz, batch.env_xyz, batch.env_species, batch.aev_nbr,
                         batch.aev_mask > 0)
    return jnp.concatenate([jnp.asarray(batch.x_lig, jnp.float32), aev], axis=1)


def _pool(h, node_graph, node_valid, n_graphs: int) -> jnp.ndarray:
    """Per graph slot, concat(max, mean) over its valid node rows; zeros for a slot without nodes.

    Segment ``n_graphs`` (where padded rows point) is computed and dropped; invalid rows are also
    kept out of the max and the mean wherever they point.
    """
    segments = n_graphs + 1
    keep = (node_valid > 0)[:, None]
    count = jax.ops.segment_sum(keep.astype(jnp.float32), node_graph, segments)[:n_graphs]
    total = jax.ops.segment_sum(jnp.where(keep, h, 0.0), node_graph, segments)[:n_graphs]
    top = jax.ops.segment_max(jnp.where(keep, h, -jnp.inf), node_graph, segments)[:n_graphs]
    top = jnp.where(count > 0, top, 0.0)
    return jnp.concatenate([top, total / jnp.maximum(count, 1.0)], axis=1)


def encode(params: dict, state: dict, batch, train: bool,
           features=aev_features) -> tuple[jnp.ndarray, jnp.ndarray, dict]:
    """Node embeddings ``(ncap, heads*hidden)`` (zero on padded rows), graph embeddings
    ``(batch, 2*heads*hidden)`` and ``state`` with the node norms updated in training mode.

    ``features(batch)`` gives the ``(ncap, 367)`` node features (default: ``aev_features``, the
    AEV on the fly). Each layer: ``gatv2_layer`` -> leaky_relu 0.01 -> masked BatchNorm over
    ``node_valid`` -> ``* node_valid``.
    """
    h = features(batch)
    valid = jnp.asarray(batch.node_valid, jnp.float32)
    layer_states = []
    for p, running in zip(params["gat"], state["gat"]):
        heads, hidden = p["att"].shape
        h = gatv2_layer(p, h, batch.nbr, batch.nbr_mask, batch.nbr_feat, heads, hidden)
        h = jax.nn.leaky_relu(h, ACT_SLOPE)
        h, running = masked_batchnorm(p["bn"], running, h, valid, train)
        h = h * valid[:, None]
        layer_states.append(running)
    graph_emb = _pool(h, batch.node_graph, valid, batch.valid.shape[0])
    return h, graph_emb, {**state, "gat": layer_states}


def readout(params: dict, state: dict, graph_emb, valid, train: bool) -> tuple[jnp.ndarray, dict]:
    """fc1 -> leaky_relu -> BatchNorm -> fc2 -> ... -> fc3 -> leaky_relu -> BatchNorm -> out.

    The dense norms take statistics over graph slots with ``valid > 0``. Returns ``(batch,)``
    predictions and ``state`` with the dense norms updated in training mode.
    """
    h = graph_emb
    state = dict(state)
    for dense, norm in zip(DENSE_NAMES, NORM_NAMES):
        h = jax.nn.leaky_relu(h @ params[dense]["w"] + params[dense]["b"], ACT_SLOPE)
        h, state[norm] = masked_batchnorm(params[norm], state[norm], h, valid, train)
    pred = h @ params["out"]["w"] + params["out"]["b"]
    return pred[:, 0], state


def predict(params: dict, state: dict, batch, train: bool,
            features=aev_features) -> tuple[jnp.ndarray, dict]:
    """``(batch,)`` predictions in standardised units and the updated ``state``; ``features`` as in
    ``encode``."""
    _, graph_emb, state = encode(params, state, batch, train, features)
    return readout(params, state, graph_emb, batch.valid, train)
