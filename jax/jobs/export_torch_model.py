"""Export a PyTorch GATv2Net state dict and its label scaler to one .npz for port_parity.py.

    python jobs/export_torch_model.py --model FILE.model --scaler FILE.pickle --out FILE.npz

Runs under the AEV-PLIG env ($AEVPLIG_PY: torch to read the state dict, sklearn to unpickle the
StandardScaler) with $TREE/src on PYTHONPATH (jobs/env.sh). Writes every state-dict tensor as a
numpy array under its key (``torch_port.load_torch_model``), plus ``y_mean`` and ``y_sd`` (the scaler's
``mean_[0]`` and ``scale_[0]``), atomically. ``torch_port.port_state_dict`` checks first that the
state dict is a GATv2Net.
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np

from aevplig_jax.torch_port import load_torch_model, port_state_dict

SCALER_KEYS = ("y_mean", "y_sd")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", type=Path, required=True, help="torch.save(state_dict) file")
    parser.add_argument("--scaler", type=Path, required=True, help="pickled StandardScaler")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    state_dict = load_torch_model(args.model)
    clash = sorted(set(state_dict) & set(SCALER_KEYS))
    if clash:
        raise ValueError(f"{args.model}: state-dict keys {clash} clash with the scaler entries")
    port_state_dict(state_dict)  # raises unless a GATv2Net state dict
    with open(args.scaler, "rb") as handle:
        scaler = pickle.load(handle)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    if mean.shape != (1,) or scale.shape != (1,):
        raise ValueError(f"{args.scaler}: expected a one-feature scaler, got mean_ {mean.shape}, "
                         f"scale_ {scale.shape}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pending = args.out.with_name(args.out.name + ".tmp")
    with open(pending, "wb") as handle:
        np.savez(handle, **state_dict, y_mean=mean[0], y_sd=scale[0])
    os.replace(pending, args.out)
    print(f"wrote {args.out}: {len(state_dict)} tensors from {args.model}, y_mean {mean[0]:.6f}, "
          f"y_sd {scale[0]:.6f} from {args.scaler}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
