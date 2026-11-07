"""AEV-PLIG training for one seed: the paper's loop (isambard/jobs/train_seed.py) in JAX.

    python -m aevplig_jax.train --store-root DIR --tag T --seed-index I --epochs 200 --out DIR
        [--aev onthefly|precomputed --aev-table FILE] [--precision default|highest]
        [--workers 15] [--prefetch 4] [--prepare-workers 12] [--limit N] [--resume]

Data: ``<store-root>/trainside`` splits ``train`` (its first ``--limit`` ids when given), ``valid``
and ``test``; labels are standardised with the mean and population sd of the training labels
(sklearn ``StandardScaler``). ``<out>/<tag>/manifest.json`` (capacities covering every seed's
training plans and the valid/test prediction batches, the scaler, counts, store path, package
hashes) is written by the first seed to start and reused by the others.

Loop: epochs numbered from 0 (``plan_epoch``'s argument and the original's selection index). Per
epoch the ``plan_epoch`` batches go through ``Prefetcher`` into ``loss_and_grad`` and ``apply``
(Adam at ``LR``), with a finite check of the loss every 50 steps; then the validation Pearson in
evaluation mode and the selection rule (``selection_step``: mean of the last ``SELECTION_WINDOW``
Pearsons beats the best so far). The loss is the weighted sum of ``TERMS``; every term is logged,
selection reads the validation Pearson only.

Run directory ``<out>/<tag>/seed-<I>/``: ``protocol.json``, ``history.jsonl`` (one row per epoch),
``checkpoints/best/state.npz`` (params, bn_state), ``checkpoints/latest/state.npz`` (also the
optimiser state and the loop state, for ``--resume``), ``selection.json`` and
``test_predictions.csv`` (``unique_id, truth, pred``); plus the parent's per-seed CSV
``<out>/predictions/<tag>_<DATASET>_test_seed<I>.csv`` (``truth, preds_<I>, preds``).

Imports jax, optax, numpy, the standard library and package modules. ``prepare_all`` forks its
pool before anything initialises JAX's backend.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import platform
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from aevplig_jax import inference, model
from aevplig_jax.batching import Capacities, StoreSource, capacities, plan_epoch, prepare_all
from aevplig_jax.constants import (AEV_DIM, BATCH, EPOCHS, LR, RDKIT_DIM, SEEDS,
                                   SELECTION_WINDOW)
from aevplig_jax.loading import DeferredScalars, LoaderConfig, LoaderTelemetry, Prefetcher
from aevplig_jax.precompute_aev import load_table
from aevplig_jax.store import Store

DATASET = "pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark"
TRAIN_STORE = "trainside"
SPLITS = ("train", "valid", "test")
INITIAL_BEST = -1.1  # the original's best_pc before the first epoch
FINITE_EVERY = 50  # steps between DeferredScalars flushes, and so between finite checks
PACKAGE = Path(__file__).resolve().parent


def mse_term(pred, batch):
    """``nn.MSELoss`` over the real batch slots."""
    return jnp.sum(batch.valid * (pred - batch.y) ** 2) / jnp.sum(batch.valid)


# name -> (weight, term(pred, batch)); the loss is the weighted sum. Later heads add entries.
TERMS: dict[str, tuple[float, Callable]] = {"mse": (1.0, mse_term)}


def precomputed_features(batch):
    """Node features of a batch whose ``x_lig`` already carries the precomputed AEV columns."""
    return jnp.asarray(batch.x_lig, jnp.float32)


FEATURES = {"onthefly": model.aev_features, "precomputed": precomputed_features}
OPTIMISERS = {"adam": optax.adam}  # name -> factory taking learning_rate


@dataclass(frozen=True)
class OptimiserConfig:
    name: str = "adam"
    learning_rate: float = LR


def build_optimiser(config: OptimiserConfig) -> optax.GradientTransformation:
    """``OPTIMISERS[name]`` with the learning rate as an injected hyperparameter: ``apply`` sets it
    from its traced ``lr`` argument, so a schedule changes it without recompiling."""
    return optax.inject_hyperparams(OPTIMISERS[config.name])(learning_rate=config.learning_rate)


def step_functions(features: Callable, optimiser: optax.GradientTransformation):
    """The jitted ``loss_and_grad``, ``apply`` and ``predict_fn``; ``TERMS`` is read when tracing."""

    def loss_fn(params, bn_state, batch):
        pred, bn_state = model.predict(params, bn_state, batch, True, features)
        values = {name: term(pred, batch) for name, (_, term) in TERMS.items()}
        loss = sum(weight * values[name] for name, (weight, _) in TERMS.items())
        return loss, (values, bn_state)

    @jax.jit
    def loss_and_grad(params, bn_state, batch):
        (loss, (values, bn_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, bn_state, batch)
        return (loss, values, bn_state), grads

    @jax.jit
    def apply(params, opt_state, grads, lr):
        opt_state = opt_state._replace(hyperparams={**opt_state.hyperparams, "learning_rate": lr})
        updates, opt_state = optimiser.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state

    return loss_and_grad, apply, inference.predict_function(features)


def selection_step(pcs: list[float], best: float,
                   window: int = SELECTION_WINDOW) -> tuple[float, bool]:
    """The original's rule after epoch ``len(pcs) - 1``: the mean of the last ``window``
    Pearsons (fewer at the start) and whether it beats ``best`` (a NaN mean never does)."""
    epoch = len(pcs) - 1
    mean = float(np.mean(pcs[max(epoch - (window - 1), 0):epoch + 1]))
    return mean, mean > best


def scaler(labels) -> tuple[float, float]:
    """``StandardScaler``'s ``mean_`` and ``scale_``: mean and population sd (1 when constant)."""
    labels = np.asarray(labels, dtype=np.float64)
    sd = float(labels.std())
    return float(labels.mean()), sd if sd > 0 else 1.0


def package_hashes() -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(PACKAGE.glob("*.py"))}


def _label(store: Store, rid: str) -> float:
    label = store.records[rid]["label"]
    return math.nan if label is None else float(label)


def split_ids(store: Store, limit: int | None) -> dict[str, list[str]]:
    """Ids of the three splits (train cut to ``limit``); raises on an empty split or a
    non-finite label."""
    ids = {split: store.split_ids(split) for split in SPLITS}
    if limit is not None:
        ids["train"] = ids["train"][:limit]
    for split, members in ids.items():
        if not members:
            raise ValueError(f"{store.path}: no {split} records")
        bad = [rid for rid in members if not math.isfinite(_label(store, rid))]
        if bad:
            raise ValueError(f"{store.path}: {len(bad)} {split} records without a finite label, "
                             f"e.g. {bad[:5]}")
    return ids


def _identity(store: Store, ids: dict[str, list[str]], args) -> dict:
    """The manifest fields every seed of a tag must share."""
    return {"store": str(store.path.resolve()), "counts": {s: len(ids[s]) for s in SPLITS},
            "epochs": args.epochs, "limit": args.limit, "seeds": list(SEEDS), "batch": BATCH}


def load_or_write_manifest(path: Path, identity: dict, build: Callable[[], dict]) -> dict:
    """The manifest at ``path`` when it matches ``identity``, else ``build()`` written atomically
    (seeds starting together write identical files). Raises when an existing one differs."""
    if path.exists():
        manifest = json.loads(path.read_text())
        found = {key: manifest.get(key) for key in identity}
        if found != identity:
            raise ValueError(f"{path} was written for {found}, this run is {identity}: "
                             "use another --tag")
        return manifest
    manifest = {**identity, **build()}
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, pending = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    with os.fdopen(handle, "w") as out:
        out.write(json.dumps(manifest, indent=1) + "\n")
    os.replace(pending, path)
    return manifest


def build_manifest(store: Store, ids: dict[str, list[str]], prepared: dict, epochs: int) -> dict:
    """Capacities over every seed's training plans and the valid/test prediction batches, the
    scaler and the package hashes."""
    train_cap = capacities(store, np.array(ids["train"]), list(SEEDS), epochs, BATCH)
    eval_cap = inference.plan_capacities(
        prepared, inference.sequential_plans(ids["valid"]) + inference.sequential_plans(ids["test"]))
    y_mean, y_sd = scaler([_label(store, rid) for rid in ids["train"]])
    return {"capacities": inference.merge_capacities(train_cap, eval_cap).to_json(),
            "y_mean": y_mean, "y_sd": y_sd, "package_sha256": package_hashes()}


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _check_finite(losses: DeferredScalars, checked: int, epoch: int) -> int:
    """Raise ``FloatingPointError`` at the first non-finite flushed loss after ``checked``;
    returns the number of losses checked."""
    for k in range(checked, len(losses.values)):
        if not math.isfinite(losses.values[k]):
            raise FloatingPointError(f"non-finite loss {losses.values[k]} at epoch {epoch}, "
                                     f"step {k} of the epoch")
    return len(losses.values)


def _write_rows(path: Path, header: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store-root", type=Path, required=True, help="holds trainside/")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed-index", type=int, required=True, choices=range(len(SEEDS)))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--aev", choices=tuple(FEATURES), default="onthefly")
    parser.add_argument("--aev-table", type=Path, help="precompute_aev.py table (--aev precomputed)")
    parser.add_argument("--precision", choices=("default", "highest"), default="default",
                        help="default matmul precision (default: TF32 on the GPU)")
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--prepare-workers", type=int, default=12,
                        help="processes preparing the records once at start-up")
    parser.add_argument("--limit", type=int, help="train on the first N training ids")
    parser.add_argument("--resume", action="store_true", help="continue from checkpoints/latest")
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    return args


def run(args: argparse.Namespace) -> dict:
    """Train one seed and predict the test split with the selected weights; returns
    ``selection.json``'s contents."""
    if args.aev == "precomputed" and args.aev_table is None:
        raise ValueError("--aev precomputed needs --aev-table (precompute_aev.py output)")
    run_dir = args.out / args.tag
    seed_dir = run_dir / f"seed-{args.seed_index}"
    best_path = seed_dir / "checkpoints" / "best" / "state.npz"
    latest_path = seed_dir / "checkpoints" / "latest" / "state.npz"
    if latest_path.exists() and not args.resume:
        raise FileExistsError(f"{seed_dir} holds a run: pass --resume to continue it, or use "
                              "another --tag")
    inference.enable_compilation_cache()

    with Store(args.store_root / TRAIN_STORE) as store:
        ids = split_ids(store, args.limit)
        prepared = prepare_all(store, [rid for split in SPLITS for rid in ids[split]],
                               args.prepare_workers)  # forks: before JAX's backend starts
        manifest = load_or_write_manifest(run_dir / "manifest.json", _identity(store, ids, args),
                                          lambda: build_manifest(store, ids, prepared, args.epochs))
        cap = Capacities.from_json(manifest["capacities"])
        y_mean, y_sd = manifest["y_mean"], manifest["y_sd"]
        table = load_table(args.aev_table, store) if args.aev == "precomputed" else None
        source = StoreSource(store, cap, y_mean, y_sd, args.workers, prepared=prepared,
                             aev_table=table)
        truth = {split: np.array([_label(store, rid) for rid in ids[split]]) for split in SPLITS}
        precision = (jax.default_matmul_precision("highest") if args.precision == "highest"
                     else contextlib.nullcontext())
        with precision:
            return _train(args, seed_dir, best_path, latest_path, ids, truth, source, manifest)


def _train(args, seed_dir: Path, best_path: Path, latest_path: Path, ids: dict, truth: dict,
           source: StoreSource, manifest: dict) -> dict:
    seed = SEEDS[args.seed_index]
    y_mean, y_sd = manifest["y_mean"], manifest["y_sd"]
    optimiser_config = OptimiserConfig()
    optimiser = build_optimiser(optimiser_config)
    loss_and_grad, apply, predict_fn = step_functions(FEATURES[args.aev], optimiser)
    config = LoaderConfig(workers=args.workers, prefetch=args.prefetch)
    history_path = seed_dir / "history.jsonl"

    if args.resume and latest_path.exists():
        templates = inference.member_templates()
        templates["opt_state"] = jax.eval_shape(optimiser.init, templates["params"])
        trees, loop = inference.load_state(latest_path, templates)
        params, bn_state, opt_state = trees["params"], trees["bn_state"], trees["opt_state"]
        start, steps, pcs, best = loop["epoch"] + 1, loop["steps"], loop["pcs"], loop["best_mean"]
        rows = loop["history"]
        print(f"resuming {seed_dir} after epoch {loop['epoch']} ({steps} steps)", flush=True)
    else:
        params = model.init_params(jax.random.key(seed))
        bn_state, opt_state = model.init_bn_state(), optimiser.init(params)
        start, steps, pcs, best, rows = 0, 0, [], INITIAL_BEST, []
    seed_dir.mkdir(parents=True, exist_ok=True)
    history_path.write_text("".join(json.dumps(row) + "\n" for row in rows))  # rows up to latest

    protocol = {"flags": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "dataset": DATASET, "seed_index": args.seed_index, "seed": seed,
                "constants": {"LR": LR, "BATCH": BATCH, "EPOCHS": EPOCHS,
                              "SELECTION_WINDOW": SELECTION_WINDOW, "SEEDS": list(SEEDS),
                              "AEV_DIM": AEV_DIM, "RDKIT_DIM": RDKIT_DIM},
                "capacities": manifest["capacities"], "y_mean": y_mean, "y_sd": y_sd,
                "counts": manifest["counts"], "store": manifest["store"],
                "optimiser": asdict(optimiser_config),
                "terms": {name: weight for name, (weight, _) in TERMS.items()},
                "package_sha256": package_hashes(),
                "versions": {"jax": jax.__version__, "optax": optax.__version__,
                             "numpy": np.__version__, "python": platform.python_version()},
                "devices": [str(device) for device in jax.devices()],
                "resumed_from_epoch": start - 1 if start else None}
    (seed_dir / "protocol.json").write_text(json.dumps(protocol, indent=1) + "\n")

    lr = jnp.asarray(LR, jnp.float32)  # constant, as the paper's Adam; a schedule would vary it
    train_ids = np.array(ids["train"])
    valid_plans = inference.sequential_plans(ids["valid"])
    for epoch in range(start, args.epochs):
        began = time.perf_counter()
        telemetry = LoaderTelemetry()
        losses = DeferredScalars(FINITE_EVERY)
        terms = {name: DeferredScalars(FINITE_EVERY) for name in TERMS}
        checked = 0
        specs = plan_epoch(train_ids, seed, epoch, BATCH)
        for _, (batch, _) in Prefetcher(source, specs, config, to_device=inference.device_batch,
                                        telemetry=telemetry):
            (loss, values, bn_state), grads = loss_and_grad(params, bn_state, batch)
            params, opt_state = apply(params, opt_state, grads, lr)
            steps += 1
            losses.add(loss)
            for name, scalars in terms.items():
                scalars.add(values[name])
            checked = _check_finite(losses, checked, epoch)
        losses.flush()
        _check_finite(losses, checked, epoch)
        jax.block_until_ready(params)
        train_seconds = time.perf_counter() - began
        wait_fraction = telemetry.summary()["wait_fraction"]  # over the training batches only

        began = time.perf_counter()
        pred = inference.predict_ids(predict_fn, [(params, bn_state)], source, valid_plans, config)
        pc = inference.pearson(truth["valid"], inference.destandardise(pred[0], y_mean, y_sd))
        valid_seconds = time.perf_counter() - began
        pcs.append(pc)
        mean, saved = selection_step(pcs, best)
        if saved:
            best = mean
            inference.save_state(best_path, {"params": params, "bn_state": bn_state},
                                 {"epoch": epoch, "valid_pearson": pc, "selection_mean": mean})
        row = {"epoch": epoch, "loss": losses.mean(),
               "terms": {name: scalars.mean() for name, scalars in terms.items()},
               "valid_pearson": _finite_or_none(pc), "selection_mean": _finite_or_none(mean),
               "saved": saved, "steps": steps, "train_seconds": train_seconds,
               "valid_seconds": valid_seconds, "wait_fraction": wait_fraction}
        rows.append(row)
        inference.save_state(latest_path,
                             {"params": params, "bn_state": bn_state, "opt_state": opt_state},
                             {"epoch": epoch, "steps": steps, "pcs": pcs, "best_mean": best,
                              "history": rows})
        with open(history_path, "a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(f"epoch {epoch}: loss {row['loss']:.4f} valid PC {pc:.4f} mean {mean:.4f}"
              f"{' saved' if saved else ''} train {train_seconds:.1f}s valid {valid_seconds:.1f}s "
              f"wait {row['wait_fraction']:.3f}", flush=True)

    return _predict_test(args, seed_dir, best_path, ids, truth, source, manifest, predict_fn,
                         config)


def _predict_test(args, seed_dir: Path, best_path: Path, ids: dict, truth: dict,
                  source: StoreSource, manifest: dict, predict_fn: Callable,
                  config: LoaderConfig) -> dict:
    """Predict the test split with the ``best`` weights; write the prediction CSVs and
    ``selection.json``."""
    if not best_path.exists():
        raise RuntimeError(f"{seed_dir}: no epoch passed the selection rule (every validation "
                           "Pearson NaN?), so there is no best checkpoint")
    params, bn_state, chosen = inference.load_member(best_path)
    pred = inference.predict_ids(predict_fn, [(params, bn_state)], source,
                                 inference.sequential_plans(ids["test"]), config)
    pred = inference.destandardise(pred[0], manifest["y_mean"], manifest["y_sd"])
    index = args.seed_index
    _write_rows(args.out / "predictions" / f"{args.tag}_{DATASET}_test_seed{index}.csv",
                ["truth", f"preds_{index}", "preds"],
                ([float(t), float(p), float(p)] for t, p in zip(truth["test"], pred)))
    _write_rows(seed_dir / "test_predictions.csv", ["unique_id", "truth", "pred"],
                ([rid, float(t), float(p)] for rid, t, p in zip(ids["test"], truth["test"], pred)))
    selection = {"epoch": chosen["epoch"], "valid_pearson": chosen["valid_pearson"],
                 "selection_mean": chosen["selection_mean"], "window": SELECTION_WINDOW,
                 "epochs": args.epochs, "test_pearson": inference.pearson(truth["test"], pred),
                 "test_rmse": inference.rmse(truth["test"], pred)}
    (seed_dir / "selection.json").write_text(json.dumps(selection, indent=1) + "\n")
    print(f"selected epoch {selection['epoch']} (valid PC {selection['valid_pearson']:.4f}, mean "
          f"{selection['selection_mean']:.4f}); test PC {selection['test_pearson']:.4f} RMSE "
          f"{selection['test_rmse']:.4f}", flush=True)
    return selection


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
