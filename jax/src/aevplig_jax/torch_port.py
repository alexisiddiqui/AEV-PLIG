"""Port a PyTorch ``GATv2Net`` state dict (model_defs.py, PyG 2.5.3) to model.py's trees.

PyTorch ``Linear`` weights are ``(out, in)`` and become ``(in, out)``; ``att``
``(1, heads, hidden)`` becomes ``(heads, hidden)``; BatchNorm weight/bias become ``scale``/``shift``
and the running statistics go to the separate BatchNorm state. ``num_batches_tracked`` is dropped.

Imports numpy and the standard library; torch only inside ``load_torch_model``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_CONV_KEYS = ("lin_l.weight", "lin_l.bias", "lin_r.weight", "lin_r.bias", "lin_edge.weight", "att",
              "bias")
_NORM_KEYS = ("weight", "bias", "running_mean", "running_var")
_DENSE = ("fc1", "fc2", "fc3", "out")
_NORMS = {"bn_connect1": "bn1", "bn_connect2": "bn2", "bn_connect3": "bn3"}


def load_torch_model(path: Path) -> dict[str, np.ndarray]:
    """A saved state dict as numpy arrays (``torch.load`` on the CPU, tensors only)."""
    import torch

    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    return {name: tensor.detach().cpu().numpy() for name, tensor in state_dict.items()}


def _expected_keys(layers: int) -> list[str]:
    keys = []
    for i in range(layers):
        keys += [f"GNN_layers.{i}.{key}" for key in _CONV_KEYS]
        keys += [f"BN_layers.{i}.module.{key}" for key in _NORM_KEYS]
    for name in _DENSE:
        keys += [f"{name}.weight", f"{name}.bias"]
    for name in _NORMS:
        keys += [f"{name}.{key}" for key in _NORM_KEYS]
    return keys


def port_state_dict(sd: dict[str, np.ndarray]) -> tuple[dict, dict]:
    """``(params, bn_state)`` in model.py's layout, float32 numpy arrays.

    The layer count is read from the ``GNN_layers.{i}`` keys. Raises ``ValueError`` listing the
    missing and unexpected keys when ``sd`` is not exactly a GATv2Net state dict (ignoring
    ``num_batches_tracked``).
    """
    sd = {name: value for name, value in sd.items() if not name.endswith(".num_batches_tracked")}
    found = {re.match(r"GNN_layers\.(\d+)\.", name) for name in sd}
    layers = 1 + max((int(match.group(1)) for match in found if match), default=-1)
    expected = _expected_keys(layers)
    missing = [name for name in expected if name not in sd]
    unexpected = sorted(set(sd) - set(expected))
    if missing or unexpected or layers == 0:
        raise ValueError(f"not a GATv2Net state dict ({layers} GNN layers): missing {missing}, "
                         f"unexpected {unexpected}")

    def array(name: str) -> np.ndarray:
        return np.asarray(sd[name], dtype=np.float32)

    def weight(name: str) -> np.ndarray:  # (out, in) -> (in, out)
        return np.ascontiguousarray(array(name).T)

    def norm(prefix: str) -> tuple[dict, dict]:
        return ({"scale": array(f"{prefix}.weight"), "shift": array(f"{prefix}.bias")},
                {"mean": array(f"{prefix}.running_mean"), "var": array(f"{prefix}.running_var")})

    params: dict = {"gat": [], "heads": {}}
    state: dict = {"gat": []}
    for i in range(layers):
        conv = f"GNN_layers.{i}"
        affine, running = norm(f"BN_layers.{i}.module")
        att = array(f"{conv}.att")
        params["gat"].append({
            "w_l": weight(f"{conv}.lin_l.weight"), "b_l": array(f"{conv}.lin_l.bias"),
            "w_r": weight(f"{conv}.lin_r.weight"), "b_r": array(f"{conv}.lin_r.bias"),
            "w_e": weight(f"{conv}.lin_edge.weight"), "att": att.reshape(att.shape[-2:]),
            "bias": array(f"{conv}.bias"), "bn": affine,
        })
        state["gat"].append(running)
    for name in _DENSE:
        params[name] = {"w": weight(f"{name}.weight"), "b": array(f"{name}.bias")}
    for torch_name, name in _NORMS.items():
        params[name], state[name] = norm(torch_name)
    return params, state
