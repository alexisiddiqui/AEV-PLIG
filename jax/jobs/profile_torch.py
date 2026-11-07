# PyTorch baseline for the JAX rewrite's profile report: the paper's training loop
# (training.py, as isambard/jobs/train_seed.py runs it) on seed index 0 of the retraining dataset.
#   cd $PARENT && AEVPLIG_REPO=<clone> $AEVPLIG_PY $TREE/jobs/profile_torch.py --out $TREE/work/output/profile/torch
# Runs from the parent's work directory (GraphDataset(root='data') resolves there) and only reads it: the
# output directory may not lie under the working directory. Phases, one model and optimiser throughout:
#   1. --epochs epochs with the paper's loaders (train: batch 128, shuffle; valid: shuffle=False), each timed
#      as training + the paper's validation predict(); per step the time to fetch a batch and move it to the
#      device ("data") is summed, so loader wait = data / training time;
#   2. torch.profiler (CPU and CUDA activities) over --profile-steps steps of the paper's loader, each step
#      split into data / forward (zero_grad, model, loss) / backward / optimiser by perf_counter with a CUDA
#      synchronise after each phase; key_averages() table -> key_averages.txt;
#   3. phase 1 again with DataLoader(num_workers=--workers, pin_memory=True, persistent_workers=True).
# Seconds per epoch = mean of train + validation over the epochs after the first (start-up, cuDNN warm-up).
# Writes <out>/profile_torch.json (read by aevplig_jax.profile, which also takes the GPU utilisation of
# each loader's "window" from <out>/gpu_util.txt) and <out>/key_averages.txt.
import os, sys
REPO = os.environ.get("AEVPLIG_REPO", "/projects/u6xq/hussain/opig-harness/src/AEV-PLIG-3ccac4b")
sys.path.insert(0, REPO)  # as train_seed.py: utils, helpers, model_defs, training from the clone
import argparse
import json
import platform
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch_geometric
from torch.profiler import ProfilerActivity, profile, record_function
from torch_geometric.loader import DataLoader

from helpers import model_dict, pearson
from training import predict  # the paper's validation pass
from utils import GraphDataset, init_weights

DATASET = "pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark"
SEEDS = [100, 123, 15, 257, 2, 2012, 3752, 350, 843, 621]
CONFIG = SimpleNamespace(activation_function="leaky_relu", head=3, hidden_dim=256)  # the paper's args
BATCH = 128
LR = 0.00012291937615434127
EPOCHS_PER_SEED = 200


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_epoch(model, device, loader, optimizer, loss_fn):
    """The paper's train() with timers: (seconds, data seconds, steps, mean loss)."""
    model.train()
    data_seconds, steps, total_loss = 0.0, 0, 0.0
    began = time.perf_counter()
    batches = iter(loader)
    while True:
        tick = time.perf_counter()
        data = next(batches, None)
        if data is None:
            break
        data = data.to(device)
        data_seconds += time.perf_counter() - tick
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(data.y)  # synchronises, as in the paper
        steps += 1
    sync(device)
    return time.perf_counter() - began, data_seconds, steps, total_loss / len(loader.dataset)


def time_epochs(model, device, train_loader, valid_loader, optimizer, loss_fn, scaler, epochs):
    """Phases 1 and 3: per-epoch rows, seconds per epoch, loader wait and the timed window."""
    rows, window = [], None
    for epoch in range(epochs):
        if epoch == min(1, epochs - 1):
            window = [time.time(), None]
        train_s, data_s, steps, loss = train_epoch(model, device, train_loader, optimizer, loss_fn)
        began = time.perf_counter()
        truth, pred = predict(model, device, valid_loader, scaler)
        valid_s = time.perf_counter() - began
        rows.append({"train_seconds": train_s, "data_seconds": data_s, "valid_seconds": valid_s,
                     "steps": steps, "loss": loss, "valid_pearson": float(pearson(truth, pred))})
        print(f"epoch {epoch}: train {train_s:.1f} s (data {data_s:.1f} s) valid {valid_s:.1f} s",
              flush=True)
    window[1] = time.time()
    steady = rows[1:] or rows
    seconds = float(np.mean([r["train_seconds"] + r["valid_seconds"] for r in steady]))
    return {"epochs": rows, "seconds_per_epoch": seconds,
            "minutes_per_seed": seconds * EPOCHS_PER_SEED / 60,
            "wait_fraction": sum(r["data_seconds"] for r in steady)
            / sum(r["train_seconds"] for r in steady),
            "window": window}


def profiled_steps(model, device, loader, optimizer, loss_fn, steps, table_path):
    """Phase 2: the split of ``steps`` training steps and the torch.profiler table."""
    seconds = {"data": 0.0, "forward": 0.0, "backward": 0.0, "optimiser": 0.0}
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    model.train()
    batches = iter(loader)

    def phase(name, work):
        tick = time.perf_counter()
        with record_function(name):
            value = work()
        sync(device)
        seconds[name] += time.perf_counter() - tick
        return value

    def fetch():
        nonlocal batches
        data = next(batches, None)
        if data is None:  # the epoch ended: start the next, as the paper's loop would
            batches = iter(loader)
            data = next(batches)
        return data.to(device)

    def forward(data):
        optimizer.zero_grad()
        return loss_fn(model(data), data.y.view(-1, 1).to(device))

    with profile(activities=activities) as prof:
        for _ in range(steps):
            data = phase("data", fetch)
            loss = phase("forward", lambda: forward(data))
            phase("backward", loss.backward)
            phase("optimiser", optimizer.step)
            loss.item()
    averages = prof.key_averages()
    table = None
    for sort_by in (("self_cuda_time_total",) if device.type == "cuda" else ()) + (
            "self_cpu_time_total",):
        try:
            table = averages.table(sort_by=sort_by, row_limit=40)
            break
        except (AttributeError, KeyError, ValueError):  # a sort key this torch does not know
            continue
    table_path.write_text(f"{table}\n")
    return {"steps": steps, "seconds": seconds, "table": table_path.name}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PyTorch AEV-PLIG loop profile")
    parser.add_argument("--out", type=Path, required=True, help="profile_torch.json goes here")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--profile-steps", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-index", type=int, default=0, choices=range(len(SEEDS)))
    args = parser.parse_args(argv)
    out, cwd = args.out.resolve(), Path.cwd().resolve()
    if out == cwd or cwd in out.parents:
        parser.error(f"--out {out} lies under the working directory {cwd}, which is read-only here")
    out.mkdir(parents=True, exist_ok=True)

    began = time.perf_counter()
    train_data = GraphDataset(root="data", dataset=DATASET + "_train", y_scaler=None)
    valid_data = GraphDataset(root="data", dataset=DATASET + "_valid", y_scaler=train_data.y_scaler)
    load_seconds = time.perf_counter() - began
    seed = SEEDS[args.seed_index]
    random.seed(seed)  # the paper's order: seeds, loaders, model, init, optimiser, device
    torch.manual_seed(int(seed))
    train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=BATCH, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_dict["GATv2Net"](node_feature_dim=train_data.num_node_features,
                                   edge_feature_dim=train_data.num_edge_features, config=CONFIG)
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0)
    loss_fn = nn.MSELoss()
    model.to(device)
    scaler = train_data.y_scaler
    print(f"[profile_torch] {len(train_data)} train / {len(valid_data)} valid graphs loaded in "
          f"{load_seconds:.1f} s; device {device}", flush=True)

    plain = {"num_workers": 0, "pin_memory": False}
    workers = {"num_workers": args.workers, "pin_memory": True, "persistent_workers": True}
    loaders = {"plain": {**plain, **time_epochs(model, device, train_loader, valid_loader,
                                                optimizer, loss_fn, scaler, args.epochs)}}
    profiler = profiled_steps(model, device, train_loader, optimizer, loss_fn, args.profile_steps,
                              out / "key_averages.txt")
    print(f"[profile_torch] profiler split over {args.profile_steps} steps: {profiler['seconds']}",
          flush=True)
    loaders["workers"] = {**workers, **time_epochs(
        model, device, DataLoader(train_data, batch_size=BATCH, shuffle=True, **workers),
        DataLoader(valid_data, batch_size=BATCH, shuffle=False, **workers), optimizer, loss_fn,
        scaler, args.epochs)}

    result = {"dataset": DATASET, "seed_index": args.seed_index, "seed": seed, "batch": BATCH,
              "epochs_timed": args.epochs, "steps_per_epoch": len(train_loader),
              "load_seconds": load_seconds, "loaders": loaders, "profiler": profiler,
              "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
              "versions": {"torch": torch.__version__,
                           "torch_geometric": torch_geometric.__version__,
                           "python": platform.python_version()},
              "repo": REPO, "cwd": str(cwd)}
    pending = out / "profile_torch.json.tmp"
    pending.write_text(json.dumps(result, indent=1) + "\n")
    os.replace(pending, out / "profile_torch.json")
    print(f"[profile_torch] wrote {out / 'profile_torch.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
