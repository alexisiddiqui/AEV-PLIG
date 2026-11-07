"""Profile of the JAX training step, and the combined PyTorch / JAX profile report.

    python -m aevplig_jax.profile --store-root DIR --manifest FILE --out DIR
        [--aev onthefly|precomputed --aev-table FILE] [--steps 300] [--trace-steps 20]
        [--warmup-steps 5] [--epochs 200] [--seed-index 0] [--workers 15] [--prefetch 4]
        [--prepare-workers 12]
    python -m aevplig_jax.profile --out DIR --seed-run SEED_DIR --seed-window START END
    python -m aevplig_jax.profile --out DIR

Profiling (first form) builds, with train.py's functions, what train.py builds: the trainside
splits, the prepared records, the manifest (``--manifest``; ``load_or_write_manifest`` writes it
when absent, and a ``train.py`` run whose ``<out>/<tag>/manifest.json`` is that file reuses it), the
``StoreSource`` (plus the ``--aev-table`` columns for ``--aev precomputed``) and the jitted
``loss_and_grad``, ``apply`` and ``predict_fn``. The batches are seed ``--seed-index``'s training
plans from epoch 0, the same in both ``--aev`` modes, through ``Prefetcher``: ``--warmup-steps``
(compilation), then ``--steps`` timed steps (``loss_and_grad`` + ``apply``, ``block_until_ready``
on the parameters) with that segment's loader telemetry, then ``--trace-steps`` under
``jax.profiler.trace`` (``<out>/trace/<aev>/``, with a perfetto file), then a validation pass
timed after a compiling one. Seconds per epoch = steps per epoch x the timed wall per step + the
validation pass. Padding over the timed batches: node rows (``1 - node_valid.mean()``), environment
rows (``1 - real rows / mcap``), AEV-list slots (``1 - real neighbours / (ncap * ka)``, i.e.
``aev_mask`` over the real rows: spread padding copies masks into padded rows). Op summary: the top
``TOP_OPS`` ops by device time from the trace's ``.xplane.pb`` when ``xprof`` (or
``tensorboard_plugin_profile``) imports, else op counts of the compiled HLO of ``loss_and_grad``.
The row goes to ``<out>/profile.json`` under ``jax.<aev>``.

``--seed-run`` records a timed ``train.py`` seed (``history.jsonl`` sums and the wall-clock window
around the run, epoch seconds) under ``seed``. Every form then rewrites ``<out>/REPORT.md`` from
``profile.json``, the PyTorch profile ``<out>/torch/profile_torch.json`` when present
(``jobs/profile_torch.py``) and the GPU utilisation in each row's window from the
``nvidia-smi dmon -s u -o DT`` logs ``<out>/gpu_util.txt`` and ``<out>/torch/gpu_util.txt``.

Imports jax, numpy, the standard library and package modules.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from aevplig_jax import inference, model, train
from aevplig_jax.batching import Capacities, Prepared, StoreSource, plan_epoch, prepare_all
from aevplig_jax.constants import BATCH, EPOCHS, LR, SEEDS
from aevplig_jax.loading import LoaderConfig, LoaderTelemetry, Prefetcher
from aevplig_jax.precompute_aev import load_table
from aevplig_jax.store import Store

PROFILE_JSON = "profile.json"
REPORT = "REPORT.md"
GPU_UTIL = "gpu_util.txt"
TORCH_DIR = "torch"
TORCH_JSON = "profile_torch.json"
TOP_OPS = 15
LABELS = {"onthefly": "JAX on-the-fly", "precomputed": "JAX precomputed"}
# design G4 / G5: minutes per seed, speed-up over PyTorch, loader wait, node padding, AEV overhead
TARGET_MINUTES, TARGET_SPEEDUP = 10, 5
TARGET_WAIT, TARGET_PADDING, TARGET_OVERHEAD = 0.05, 0.25, 0.1


# ----------------------------------------------------------------------------- measurement

def step_plans(train_ids: np.ndarray, seed: int, count: int) -> list[np.ndarray]:
    """The first ``count`` batches of ``plan_epoch`` over epochs 0, 1, ..."""
    specs, epoch = [], 0
    while len(specs) < count:
        specs += plan_epoch(train_ids, seed, epoch, BATCH)
        epoch += 1
    return specs[:count]


def padding_fractions(prepared: dict[str, Prepared], specs, cap: Capacities) -> dict[str, float]:
    """Mean padded share over ``specs`` of the node rows, environment rows and AEV-list slots."""
    node = env = aev = 0.0
    for spec in specs:
        items = [prepared[str(rid)] for rid in spec]
        node += sum(len(item.xyz) for item in items) / cap.ncap
        env += sum(len(item.env_xyz) for item in items) / cap.mcap
        aev += sum(int(item.aev_count.sum()) for item in items) / (cap.ncap * cap.ka)
    n = len(specs)
    return {"node": 1.0 - node / n, "env": 1.0 - env / n, "aev": 1.0 - aev / n}


def _segment(step, state, source, specs, config: LoaderConfig) -> tuple[tuple, dict]:
    """Run ``step`` over the batches of ``specs``; per-step seconds, wall, loader telemetry and the
    wall-clock window (epoch seconds)."""
    telemetry, times = LoaderTelemetry(), []
    start, began = time.time(), time.perf_counter()
    for _, (batch, _) in Prefetcher(source, specs, config, to_device=inference.device_batch,
                                    telemetry=telemetry):
        tick = time.perf_counter()
        state = step(state, batch)
        times.append(time.perf_counter() - tick)
    wall = time.perf_counter() - began
    return state, {"times": times, "wall": wall, "loader": telemetry.summary(),
                   "window": [start, time.time()]}


def _opcode(rest: str) -> str | None:
    """The opcode of an HLO instruction's right-hand side (shape skipped); fusions as
    ``fusion:<kind>[:<backend kind>]``, custom calls as ``custom-call:<target>``."""
    if rest.startswith("("):  # tuple shape
        depth = 0
        for end, char in enumerate(rest):
            depth += {"(": 1, ")": -1}.get(char, 0)
            if depth == 0:
                break
        rest = rest[end + 1:]
    else:
        rest = rest.split(" ", 1)[1] if " " in rest else ""
    match = re.match(r"\s*([\w\-]+)\(", rest)
    if match is None:
        return None
    op = match.group(1)
    if op == "fusion":  # kind, and the backend's fusion kind (e.g. __triton_gemm) when it has one
        op += "".join(f":{found.group(1)}" for pattern in (r"kind=(\w+)", r'"kind":"([^"]+)"')
                      if (found := re.search(pattern, rest)))
    elif op == "custom-call" and (found := re.search(r'custom_call_target="([^"]+)"', rest)):
        op += f":{found.group(1)}"
    return op


_COMPUTATION = re.compile(r"^(?:ENTRY\s+)?%?([\w.\-]+)\s.*\{\s*$")
_INSTRUCTION = re.compile(r"^\s+(?:ROOT\s+)?%?[\w.\-]+\s*=\s*(.*)$")
_BOOKKEEPING = {"parameter", "constant", "tuple", "get-tuple-element", "bitcast"}


def hlo_op_counts(text: str) -> Counter:
    """Op counts of the computations that launch work (the entry, loop bodies, branches) of a
    compiled HLO module's text; the bodies of fusions and reducers (``calls=`` / ``to_apply=``)
    and the ops that launch nothing (``_BOOKKEEPING``) are not counted."""
    computations: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        header = _COMPUTATION.match(line)
        if header and not line[:1].isspace():
            current = computations.setdefault(header.group(1), [])
        elif line.strip() == "}":
            current = None
        elif current is not None and (instruction := _INSTRUCTION.match(line)):
            current.append(instruction.group(1))
    nested = {name for body in computations.values() for rest in body
              for name in re.findall(r"(?:calls|to_apply)=%?([\w.\-]+)", rest)}
    return Counter(op for name, body in computations.items() if name not in nested
                   for op in map(_opcode, body) if op and op not in _BOOKKEEPING)


def _xprof_rows(trace_dir: Path, top: int) -> list | None:
    """``[op, device self time]`` rows from the trace's ``.xplane.pb`` through xprof's
    ``framework_op_stats`` tool; None when neither xprof nor tensorboard_plugin_profile imports or
    the conversion fails (neither is in the training venvs, so this path is untested)."""
    try:
        from xprof.convert import raw_to_tool_data
    except ImportError:
        try:
            from tensorboard_plugin_profile.convert import raw_to_tool_data
        except ImportError:
            return None
    try:
        paths = [str(path) for path in sorted(trace_dir.rglob("*.xplane.pb"))]
        data, _ = raw_to_tool_data.xspace_to_tool_data(paths, "framework_op_stats", {})
        tables = json.loads(data)
        rows = []
        for table in tables if isinstance(tables, list) else [tables]:
            ids = [column.get("id") for column in table["cols"]]
            op, self_time = ids.index("operation"), ids.index("total_self_time")
            where = ids.index("host_or_device") if "host_or_device" in ids else None
            for row in table["rows"]:
                cells = [cell.get("v") if cell else None for cell in row["c"]]
                if where is None or cells[where] == "Device":
                    rows.append([str(cells[op]), float(cells[self_time])])
        return sorted(rows, key=lambda row: -row[1])[:top] or None
    except Exception:  # an unfamiliar xprof output: fall back to the HLO counts
        return None


def op_summary(trace_dir: Path | None, hlo_text: str, top: int = TOP_OPS) -> dict:
    rows = _xprof_rows(trace_dir, top) if trace_dir is not None else None
    if rows:
        return {"source": "xprof framework_op_stats over the traced steps (device self time, us)",
                "rows": rows}
    return {"source": "op counts of the compiled HLO of loss_and_grad (no times: xprof is not "
                      "installed)",
            "rows": [list(item) for item in hlo_op_counts(hlo_text).most_common(top)]}


def _measure(args, source: StoreSource, specs: list, valid_ids: list[str], n_train: int,
             trace_dir: Path) -> dict:
    optimiser = train.build_optimiser(train.OptimiserConfig())
    loss_and_grad, apply, predict_fn = train.step_functions(train.FEATURES[args.aev], optimiser)
    params = model.init_params(jax.random.key(SEEDS[args.seed_index]))
    state = (params, model.init_bn_state(), optimiser.init(params))
    lr = jnp.asarray(LR, jnp.float32)
    config = LoaderConfig(workers=args.workers, prefetch=args.prefetch)

    def step(state, batch):
        params, bn_state, opt_state = state
        (_, _, bn_state), grads = loss_and_grad(params, bn_state, batch)
        params, opt_state = apply(params, opt_state, grads, lr)
        jax.block_until_ready(params)
        return params, bn_state, opt_state

    warm_end, timed_end = args.warmup_steps, args.warmup_steps + args.steps
    state, warm = _segment(step, state, source, specs[:warm_end], config)
    state, timed = _segment(step, state, source, specs[warm_end:timed_end], config)
    trace = {"dir": str(trace_dir), "steps": args.trace_steps, "error": None}
    if args.trace_steps:
        shutil.rmtree(trace_dir, ignore_errors=True)
        try:  # a profiler failure (e.g. no CUPTI) must not lose the timings
            with jax.profiler.trace(str(trace_dir), create_perfetto_trace=True):
                state, _ = _segment(step, state, source, specs[timed_end:], config)
        except Exception as error:
            trace["error"] = f"{type(error).__name__}: {error}"

    member = [(state[0], state[1])]
    valid_plans = inference.sequential_plans(valid_ids)
    for _ in range(2):  # the first pass compiles predict_fn
        began = time.perf_counter()
        inference.predict_ids(predict_fn, member, source, valid_plans, config)
        valid_seconds = time.perf_counter() - began

    batch, _ = inference.device_batch(source.load(specs[0]))
    hlo = loss_and_grad.lower(state[0], state[1], batch).compile().as_text()
    times = np.array(timed["times"])
    steps_per_epoch = math.ceil(n_train / BATCH)
    seconds_per_epoch = steps_per_epoch * timed["wall"] / len(times) + valid_seconds
    return {"aev": args.aev, "seed_index": args.seed_index, "warmup_steps": args.warmup_steps,
            "steps": args.steps, "trace_steps": args.trace_steps,
            "first_step_seconds": warm["times"][0] if warm["times"] else None,
            "step_seconds": {"mean": float(times.mean()), "p50": float(np.percentile(times, 50)),
                             "p95": float(np.percentile(times, 95)), "min": float(times.min()),
                             "max": float(times.max())},
            "segment_seconds": timed["wall"], "valid_seconds": valid_seconds,
            "steps_per_epoch": steps_per_epoch, "epochs": args.epochs,
            "seconds_per_epoch": seconds_per_epoch,
            "minutes_per_seed": args.epochs * seconds_per_epoch / 60,
            "wait_fraction": timed["loader"]["wait_fraction"], "loader": timed["loader"],
            "window": timed["window"], "trace": trace,
            "op_summary": op_summary(trace_dir if args.trace_steps and not trace["error"] else None,
                                     hlo),
            "devices": [str(device) for device in jax.devices()],
            "versions": {"jax": jax.__version__, "numpy": np.__version__,
                         "python": platform.python_version()}}


def profile_step(args) -> dict:
    """Measure one ``--aev`` mode; the ``profile.json`` row."""
    inference.enable_compilation_cache()
    with Store(args.store_root / train.TRAIN_STORE) as store:
        ids = train.split_ids(store, None)
        prepared = prepare_all(store, [rid for split in train.SPLITS for rid in ids[split]],
                               args.prepare_workers)  # forks: before JAX's backend starts
        identity = train._identity(store, ids, argparse.Namespace(epochs=args.epochs, limit=None))
        manifest = train.load_or_write_manifest(
            args.manifest, identity,
            lambda: train.build_manifest(store, ids, prepared, args.epochs))
        cap = Capacities.from_json(manifest["capacities"])
        table = load_table(args.aev_table, store) if args.aev == "precomputed" else None
        source = StoreSource(store, cap, manifest["y_mean"], manifest["y_sd"], args.workers,
                             prepared=prepared, aev_table=table)
        specs = step_plans(np.array(ids["train"]), SEEDS[args.seed_index],
                           args.warmup_steps + args.steps + args.trace_steps)
        row = _measure(args, source, specs, ids["valid"], len(ids["train"]),
                       args.out / "trace" / args.aev)
        row["padding"] = padding_fractions(prepared, specs[args.warmup_steps:
                                                           args.warmup_steps + args.steps], cap)
        row["capacities"] = cap.to_json()
        row["manifest"] = str(args.manifest)
    return row


def seed_row(seed_dir: Path, window: list[float]) -> dict:
    """A timed ``train.py`` seed: wall clock of ``window`` and the sums of its ``history.jsonl``."""
    rows = [json.loads(line) for line in (seed_dir / "history.jsonl").read_text().splitlines()]
    waits = [row["wait_fraction"] for row in rows]
    return {"run": str(seed_dir), "window": list(window), "wall_seconds": window[1] - window[0],
            "epochs": len(rows), "train_seconds": sum(row["train_seconds"] for row in rows),
            "valid_seconds": sum(row["valid_seconds"] for row in rows),
            "wait_fraction": float(np.mean(waits)) if waits else None}


# ----------------------------------------------------------------------------- report

def gpu_utilisation(path: Path, window) -> float | None:
    """Mean SM utilisation (a fraction) of the ``nvidia-smi dmon -s u -o DT`` samples of ``path``
    stamped within ``window`` (epoch seconds, local time); None without a file, window or sample."""
    path = Path(path)
    if not window or not path.is_file():
        return None
    columns, values = None, []
    for line in path.read_text().splitlines():
        fields = line.lstrip("#").split()
        if line.startswith("#"):
            if columns is None and "sm" in fields and "Time" in fields:
                columns = fields
            continue
        if columns is None or len(fields) != len(columns):
            continue
        sample = dict(zip(columns, fields))
        try:
            stamp = time.mktime(time.strptime(sample["Date"] + sample["Time"], "%Y%m%d%H:%M:%S"))
            utilisation = float(sample["sm"])
        except (KeyError, ValueError):
            continue
        if math.floor(window[0]) <= stamp <= math.ceil(window[1]):  # dmon stamps whole seconds
            values.append(utilisation)
    return float(np.mean(values)) / 100.0 if values else None


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, pending = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    with os.fdopen(handle, "w") as out:
        out.write(json.dumps(value, indent=1) + "\n")
    os.replace(pending, path)


def collect(out: Path) -> dict:
    """``profile.json`` with the PyTorch profile under ``torch`` (when present) and ``gpu_util``
    on every row that has a window."""
    profile = _read_json(out / PROFILE_JSON)
    torch_profile = _read_json(out / TORCH_DIR / TORCH_JSON)
    if torch_profile:
        profile["torch"] = torch_profile
    seed = [profile["seed"]] if "seed" in profile else []
    for row in [*profile.get("jax", {}).values(), *seed]:
        row["gpu_util"] = gpu_utilisation(out / GPU_UTIL, row.get("window"))
    for row in profile.get("torch", {}).get("loaders", {}).values():
        row["gpu_util"] = gpu_utilisation(out / TORCH_DIR / GPU_UTIL, row.get("window"))
    return profile


def _pct(value) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _num(value, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _overhead(jax_rows: dict) -> float | None:
    """On-the-fly over precomputed mean step time, minus one; None unless both were measured."""
    if not {"onthefly", "precomputed"} <= set(jax_rows):
        return None
    return (jax_rows["onthefly"]["step_seconds"]["mean"]
            / jax_rows["precomputed"]["step_seconds"]["mean"] - 1.0)


def _targets(profile: dict, overhead: float | None) -> list[str]:
    """The design's G4 / G5 targets against the full seed when it was timed (else the on-the-fly
    estimate) and the on-the-fly profile."""
    fly = profile.get("jax", {}).get("onthefly", {})
    seed = profile.get("seed")
    minutes = seed["wall_seconds"] / 60 if seed else fly.get("minutes_per_seed")
    torch_minutes = profile.get("torch", {}).get("loaders", {}).get("plain", {}).get(
        "minutes_per_seed")
    speedup = torch_minutes / minutes if torch_minutes and minutes else None
    wait, padding = fly.get("wait_fraction"), fly.get("padding", {}).get("node")
    checks = [(f"seed at most {TARGET_MINUTES} min (200 epochs with validation)",
               f"{_num(minutes)} min ({'measured' if seed else 'estimated'})", minutes,
               minutes is not None and minutes <= TARGET_MINUTES),
              (f"at least {TARGET_SPEEDUP}x the PyTorch loop",
               "n/a" if speedup is None else f"{speedup:.1f}x", speedup,
               speedup is not None and speedup >= TARGET_SPEEDUP),
              (f"loader wait under {_pct(TARGET_WAIT)}", _pct(wait), wait,
               wait is not None and wait < TARGET_WAIT),
              (f"node padding under {_pct(TARGET_PADDING)}", _pct(padding), padding,
               padding is not None and padding < TARGET_PADDING),
              (f"on-the-fly overhead under {_pct(TARGET_OVERHEAD)}", _pct(overhead), overhead,
               overhead is not None and overhead < TARGET_OVERHEAD)]
    return ["| target | measured | met |", "| --- | --- | --- |",
            *[f"| {name} | {shown} | {'n/a' if value is None else ('yes' if met else 'no')} |"
              for name, shown, value, met in checks]]


def render_report(profile: dict) -> str:
    """REPORT.md from a ``collect`` result: the four-row table, the on-the-fly overhead, the
    targets, the JAX step details and op summaries, the PyTorch profiler split and the full seed."""
    jax_rows = profile.get("jax", {})
    loaders = profile.get("torch", {}).get("loaders", {})
    table = [("PyTorch", loaders.get("plain")), ("PyTorch workers", loaders.get("workers")),
             *[(LABELS[mode], jax_rows.get(mode)) for mode in ("precomputed", "onthefly")]]
    lines = ["# AEV-PLIG loop profile", "",
             "PyTorch: the paper's loop (`jobs/profile_torch.py`), seconds per epoch from the "
             "timed epochs (train + validation) after the first; workers: `num_workers=8, "
             "pin_memory=True, persistent_workers=True`. JAX: `profile.py`, seconds per epoch "
             "estimated from the timed steps (steps per epoch x wall per step + one validation "
             "pass, start-up excluded); the full seed below is measured end to end. GPU "
             "utilisation: mean `nvidia-smi dmon` SM % over each row's timed window (5 s samples). "
             "Padding: share of padded node rows, environment rows and AEV-list slots.", "",
             "| loop | s / epoch | min / seed (200 epochs) | GPU util | loader wait | node padding "
             "| env padding | AEV-slot padding |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, row in table:
        row = row or {}
        padding = row.get("padding", {})
        lines.append(f"| {name} | {_num(row.get('seconds_per_epoch'))} | "
                     f"{_num(row.get('minutes_per_seed'))} | {_pct(row.get('gpu_util'))} | "
                     f"{_pct(row.get('wait_fraction'))} | {_pct(padding.get('node'))} | "
                     f"{_pct(padding.get('env'))} | {_pct(padding.get('aev'))} |")
    overhead = _overhead(jax_rows)
    lines += ["", f"on-the-fly overhead: {_pct(overhead)}"
              + (f" (mean step {1e3 * jax_rows['onthefly']['step_seconds']['mean']:.2f} ms on the "
                 f"fly vs {1e3 * jax_rows['precomputed']['step_seconds']['mean']:.2f} ms "
                 "precomputed, same batches)" if overhead is not None else ""), "",
              "The precomputed A/B measures compute only: the environment arrays are still "
              "assembled and transferred in both modes, and the precomputed batches also carry "
              "the 352 AEV columns.", "", "## Targets", "", *_targets(profile, overhead), ""]

    seed = profile.get("seed")
    if seed:
        per_epoch = (seed["train_seconds"] + seed["valid_seconds"]) / max(seed["epochs"], 1)
        lines += ["## Full seed (measured)", "",
                  f"`{seed['run']}`: {seed['wall_seconds'] / 60:.1f} min wall end to end "
                  f"(start-up, {seed['epochs']} epochs with validation, test prediction); "
                  f"history: train {seed['train_seconds']:.0f} s + validation "
                  f"{seed['valid_seconds']:.0f} s ({per_epoch:.2f} s per epoch); mean loader wait "
                  f"{_pct(seed['wait_fraction'])}; GPU utilisation {_pct(seed.get('gpu_util'))}.",
                  ""]

    for mode in ("onthefly", "precomputed"):
        row = jax_rows.get(mode)
        if not row:
            continue
        step, cap = row["step_seconds"], row.get("capacities", {})
        trace = row.get("trace") or {}
        lines += [f"## {LABELS[mode]}", "",
                  f"Step (`loss_and_grad` + `apply`, `block_until_ready`): mean "
                  f"{1e3 * step['mean']:.2f} ms, p50 {1e3 * step['p50']:.2f} ms, p95 "
                  f"{1e3 * step['p95']:.2f} ms over {row['steps']} steps after "
                  f"{row.get('warmup_steps')} warm-up steps (first step "
                  f"{_num(row.get('first_step_seconds'))} s with compilation); timed segment "
                  f"{row['segment_seconds']:.2f} s; validation pass {row['valid_seconds']:.2f} s; "
                  f"{row['steps_per_epoch']} steps per epoch; capacities ncap {cap.get('ncap')}, "
                  f"mcap {cap.get('mcap')}, kb {cap.get('kb')}, ka {cap.get('ka')}; trace of "
                  f"{row.get('trace_steps')} steps in `{trace.get('dir')}` (xplane.pb and "
                  "perfetto_trace.json.gz)"
                  + (f" FAILED: {trace['error']}" if trace.get("error") else "") + ".", "",
                  f"Top ops: {row['op_summary']['source']}.", "", "| op | value |", "| --- | --- |",
                  *[f"| {op} | {value:g} |" for op, value in row["op_summary"]["rows"]], ""]

    profiler = profile.get("torch", {}).get("profiler")
    if profiler:
        total = sum(profiler["seconds"].values()) or 1.0
        lines += [f"## PyTorch profiler ({profiler['steps']} steps, the paper's loader)", "",
                  "| phase | seconds | share |", "| --- | --- | --- |",
                  *[f"| {phase} | {seconds:.2f} | {_pct(seconds / total)} |"
                    for phase, seconds in profiler["seconds"].items()], "",
                  f"`torch.profiler` table: `{TORCH_DIR}/{profiler['table']}`.", ""]
    return "\n".join(lines)


def write_report(out: Path) -> Path:
    path = out / REPORT
    path.write_text(render_report(collect(out)))
    return path


# ----------------------------------------------------------------------------- command line

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True,
                        help="profile directory: profile.json, REPORT.md, trace/")
    parser.add_argument("--store-root", type=Path, help="holds trainside/ (profile a step)")
    parser.add_argument("--manifest", type=Path,
                        help="train.py manifest.json; written when absent")
    parser.add_argument("--aev", choices=tuple(train.FEATURES), default="onthefly")
    parser.add_argument("--aev-table", type=Path,
                        help="precompute_aev.py table (--aev precomputed)")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--trace-steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="epochs per seed: the manifest's plan horizon and the estimate")
    parser.add_argument("--seed-index", type=int, default=0, choices=range(len(SEEDS)))
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--prepare-workers", type=int, default=12)
    parser.add_argument("--seed-run", type=Path,
                        help="record this timed train.py seed directory instead of profiling")
    parser.add_argument("--seed-window", type=float, nargs=2, metavar=("START", "END"),
                        help="epoch seconds before and after the --seed-run training")
    args = parser.parse_args(argv)
    if args.seed_run is not None and args.seed_window is None:
        parser.error("--seed-run needs --seed-window START END")
    if args.store_root is not None:
        if args.manifest is None:
            parser.error("profiling needs --manifest")
        if args.aev == "precomputed" and args.aev_table is None:
            parser.error("--aev precomputed needs --aev-table (precompute_aev.py output)")
        if args.steps < 1 or args.warmup_steps < 1 or args.trace_steps < 0:
            parser.error("--steps and --warmup-steps must be at least 1, --trace-steps at least 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.store_root is not None or args.seed_run is not None:
        row = profile_step(args) if args.store_root is not None else None
        profile = _read_json(args.out / PROFILE_JSON)  # after the measurement: rows of other runs
        if row is not None:
            profile.setdefault("jax", {})[args.aev] = row
        if args.seed_run is not None:
            profile["seed"] = seed_row(args.seed_run, args.seed_window)
        _write_json(args.out / PROFILE_JSON, profile)
    print(f"wrote {write_report(args.out)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
