"""Profile of the JAX training step, and the combined PyTorch / JAX profile report.

    python -m aevplig_jax.profile --store-root DIR --manifest FILE --out DIR
        [--aev onthefly|precomputed --aev-table FILE] [--steps 300] [--trace-steps 20]
        [--warmup-steps 5] [--epochs 200] [--seed-index 0] [--workers 15] [--prefetch 4]
        [--prepare-workers 12]
    python -m aevplig_jax.profile --out DIR --seed-run SEED_DIR --seed-window START END
    python -m aevplig_jax.profile --out DIR
    python -m aevplig_jax.profile --trace-summary TRACE_DIR --out DIR [--trace-steps 20]

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
``aev_mask`` over the real rows: spread padding copies masks into padded rows), GAT-list slots
(``gat``: ``1 - real slots / (ncap * kb)``, a real row using its in-degree plus the self-loop; the
share of the per-layer ``(ncap, kb, heads, hidden)`` gather that is masked) and the GAT slot fill
``gat_fill`` (``nbr_mask.mean()`` over the real rows). Kernel summary (``trace_kernels``): the
perfetto trace's GPU kernel events (``ph == "X"`` on the stream lines of the ``/device:GPU``
processes; derived lines such as "XLA Ops" repeat kernels and are left out), time summed by kernel
name, by family (instance numbers stripped) and by HLO module over the trace window, with the GPU
busy share of that window (union of the kernel intervals); ``<out>/trace/<aev>/kernels.csv`` holds
every kernel and the row the top ``TOP_KERNELS``. Op summary, the static fallback: the top
``TOP_OPS`` ops by device time from the trace's ``.xplane.pb`` when ``xprof`` (or
``tensorboard_plugin_profile``) imports, else op counts of the compiled HLO of ``loss_and_grad``.
The row goes to ``<out>/profile.json`` under ``jax.<aev>``.

``--seed-run`` records a timed ``train.py`` seed (``history.jsonl`` sums and the wall-clock window
around the run, epoch seconds) under ``seed``. Every form then rewrites ``<out>/REPORT.md`` from
``profile.json``, the PyTorch profile ``<out>/torch/profile_torch.json`` when present
(``jobs/profile_torch.py``) and the GPU utilisation in each row's window from the
``nvidia-smi dmon -s u -o DT`` logs ``<out>/gpu_util.txt`` and ``<out>/torch/gpu_util.txt``. The
report's conclusion on where the time goes is ``profile.json``'s ``diagnosis`` (written by hand
from the measured tables, then re-rendered with the third form); until then it reads "pending the
per-kernel table".

``--trace-summary`` summarises an existing trace directory (``--trace-steps`` traced steps) into
``<out>/kernels.csv`` and ``<out>/kernels.json`` and prints the tables; no report is written.

Imports jax, numpy, the standard library and package modules.
"""

from __future__ import annotations

import argparse
import csv
import gzip
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
TOP_KERNELS = 15
PERFETTO = "perfetto_trace.json.gz"  # jax.profiler.trace(create_perfetto_trace=True)
KERNELS_CSV = "kernels.csv"
KERNELS_JSON = "kernels.json"
PENDING = "pending the per-kernel table"
# GPU trace lines derived from the kernels (they repeat kernel time) or not kernels at all
DERIVED_LINES = ("XLA Modules", "XLA Ops", "XLA TraceMe", "Steps", "Framework", "TensorFlow",
                 "Source code", "Launch Stats")
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
    """Mean padded share over ``specs`` of the node rows, environment rows, AEV-list slots and
    GAT-list slots (``gat``), and the GAT slot fill of the real rows (``gat_fill``,
    ``nbr_mask.mean()`` over them: in-degree plus the self-loop over ``kb``)."""
    node = env = aev = gat = fill = 0.0
    for spec in specs:
        items = [prepared[str(rid)] for rid in spec]
        rows = sum(len(item.xyz) for item in items)
        slots = sum(int(item.degree.sum()) + len(item.degree) for item in items)
        node += rows / cap.ncap
        env += sum(len(item.env_xyz) for item in items) / cap.mcap
        aev += sum(int(item.aev_count.sum()) for item in items) / (cap.ncap * cap.ka)
        gat += slots / (cap.ncap * cap.kb)
        fill += slots / (max(rows, 1) * cap.kb)
    n = len(specs)
    return {"node": 1.0 - node / n, "env": 1.0 - env / n, "aev": 1.0 - aev / n,
            "gat": 1.0 - gat / n, "gat_fill": fill / n}


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
# launch nothing themselves (a call's body is counted instead)
_BOOKKEEPING = {"parameter", "constant", "tuple", "get-tuple-element", "bitcast", "call"}


def hlo_op_counts(text: str) -> Counter:
    """Op counts of the computations that launch work (the entry, loop bodies, branches, ``call``
    bodies such as XLA GPU command buffers) of a compiled HLO module's text. Not counted: the
    bodies of fusions (``calls=``) and of reducers (``to_apply=`` of any op but ``call``: reduce,
    scatter, sort, all-reduce, map), and the ops that launch nothing (``_BOOKKEEPING``)."""
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
    nested = set()
    for body in computations.values():
        for rest in body:
            nested.update(re.findall(r"\bcalls=%?([\w.\-]+)", rest))
            if _opcode(rest) != "call":
                nested.update(re.findall(r"\bto_apply=%?([\w.\-]+)", rest))
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


def find_trace(trace_dir: Path) -> Path | None:
    """The newest ``perfetto_trace.json.gz`` under ``trace_dir`` (or ``trace_dir`` itself when it
    is a file); None when there is none."""
    trace_dir = Path(trace_dir)
    if trace_dir.is_file():
        return trace_dir
    found = sorted(trace_dir.rglob(PERFETTO), key=lambda path: path.stat().st_mtime)
    return found[-1] if found else None


def kernel_events(trace: dict | list) -> tuple[list[dict], list[str]]:
    """The GPU kernel events of a Chrome/perfetto trace (``traceEvents``) and the lines they come
    from. Device processes: ``process_name`` containing ``/device:GPU``. Lines: the device threads
    named ``Stream ...`` (CUPTI's kernel and memcpy lines); when a device process has none, its
    threads other than ``DERIVED_LINES``. Events: ``ph == "X"`` on those lines."""
    events = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
    processes, threads = {}, {}
    for event in events:
        if event.get("ph") != "M":
            continue
        name = (event.get("args") or {}).get("name", "")
        if event.get("name") == "process_name":
            processes[event.get("pid")] = str(name)
        elif event.get("name") == "thread_name":
            threads[(event.get("pid"), event.get("tid"))] = str(name)
    devices = ({pid for pid, name in processes.items() if "/device:GPU" in name}
               or {pid for pid, name in processes.items() if "gpu" in name.lower()})
    lines = set()
    for pid in devices:
        own = {key: name for key, name in threads.items() if key[0] == pid}
        streams = {key for key, name in own.items() if name.startswith("Stream")}
        lines |= streams or {key for key, name in own.items() if not name.startswith(DERIVED_LINES)}
    picked = [event for event in events if event.get("ph") == "X" and "dur" in event
              and (event.get("pid"), event.get("tid")) in lines]
    used = sorted({(event["pid"], event.get("tid")) for event in picked}, key=str)
    return picked, [f"{processes[pid]} {threads.get((pid, tid), tid)}" for pid, tid in used]


def _family(name: str) -> str:
    """A kernel name without trailing instance numbers (``loop_add_fusion_12`` and
    ``input_reduce_fusion.3`` -> their stems)."""
    return re.sub(r"(?:[._]\d+)+$", "", name) or name


def _table(totals: dict[str, list], total_us: float, top: int | None) -> list[list]:
    """``[name, count, total ms, share of kernel time]`` rows, longest first."""
    rows = sorted(totals.items(), key=lambda item: (-item[1][1], item[0]))
    return [[name, count, us / 1e3, us / total_us if total_us else 0.0]
            for name, (count, us) in rows[:top]]


def summarise_kernels(events: list[dict], steps: int | None, top: int = TOP_KERNELS
                      ) -> tuple[dict, list[list]]:
    """The kernel summary of ``events`` (``ts``, ``dur`` in microseconds) and every kernel's
    ``[name, count, total ms, share]`` row. Window: first kernel start to last kernel end; busy:
    the union of the kernel intervals (streams may overlap); shares are of the summed kernel time."""
    by_name, by_family, by_module = {}, {}, {}
    intervals = []
    for event in events:
        name, start, dur = str(event.get("name", "")), float(event["ts"]), float(event["dur"])
        intervals.append((start, start + dur))
        module = (event.get("args") or {}).get("hlo_module")
        for totals, key in ((by_name, name), (by_family, _family(name)), (by_module, module)):
            if key is not None:
                entry = totals.setdefault(str(key), [0, 0.0])
                entry[0] += 1
                entry[1] += dur
    total = sum(entry[1] for entry in by_name.values())
    busy, reach = 0.0, -math.inf
    for start, end in sorted(intervals):
        if end > reach:
            busy += end - max(start, reach)
            reach = end
    window = (max(end for _, end in intervals) - min(start for start, _ in intervals)
              if intervals else 0.0)
    per_step = ({"kernels": len(events) / steps, "window_ms": window / 1e3 / steps,
                 "kernel_ms": total / 1e3 / steps} if steps else None)
    summary = {"events": len(events), "steps": steps, "window_ms": window / 1e3,
               "kernel_ms": total / 1e3, "busy_ms": busy / 1e3,
               "busy_fraction": busy / window if window else None, "idle_ms": (window - busy) / 1e3,
               "per_step": per_step, "top": _table(by_name, total, top),
               "families": _table(by_family, total, top),
               "modules": _table(by_module, total, None)}
    return summary, _table(by_name, total, None)


def trace_kernels(trace_dir: Path, out_dir: Path, steps: int | None) -> dict:
    """``summarise_kernels`` of the perfetto trace under ``trace_dir``, with ``source``, ``trace``,
    ``lines`` and ``csv`` (``<out_dir>/kernels.csv``, every kernel); ``{"error": ...}`` when there is
    no trace or it holds no GPU kernel events."""
    path = find_trace(trace_dir)
    if path is None:
        return {"error": f"no {PERFETTO} under {trace_dir}"}
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt") as handle:
            trace = json.load(handle)
        events, lines = kernel_events(trace)
    except (OSError, ValueError, AttributeError) as error:  # keep the timings of the row
        return {"error": f"unreadable {path}: {type(error).__name__}: {error}", "trace": str(path)}
    if not events:
        listed = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
        names = sorted({str((event.get("args") or {}).get("name")) for event in listed
                        if event.get("ph") == "M" and event.get("name") == "process_name"})
        return {"error": f"no GPU kernel events in {path} (processes: {names[:12]})",
                "trace": str(path)}
    summary, rows = summarise_kernels(events, steps)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / KERNELS_CSV, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "count", "total_ms", "share"])
        writer.writerows(rows)
    return {"source": "GPU kernel events of the perfetto trace (ph X on the device stream lines; "
                      "dur in us), summed over the traced steps",
            "trace": str(path), "lines": lines, "csv": str(out_dir / KERNELS_CSV), **summary}


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
    trace = {"dir": str(trace_dir), "steps": args.trace_steps, "error": None,
             "xla_flags": os.environ.get("XLA_FLAGS", "")}
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
            "kernels": (trace_kernels(trace_dir, trace_dir, args.trace_steps)
                        if args.trace_steps and not trace["error"]
                        else {"error": trace["error"] or "no trace (--trace-steps 0)"}),
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


def _cell(text: str, width: int = 100) -> str:
    """A kernel name for a markdown table cell: pipes escaped, long names cut (kernels.csv has
    them whole)."""
    text = text.replace("|", "\\|")
    return text if len(text) <= width else text[:width - 3] + "..."


def kernel_lines(kernels: dict | None) -> list[str]:
    """The measured per-kernel section of a JAX row (``trace_kernels``), or why there is none."""
    if not kernels:
        return ["Kernel time: not measured (no trace summary in this row; `profile.py "
                "--trace-summary` reads an existing trace).", ""]
    if kernels.get("error"):
        return [f"Kernel time: no table ({kernels['error']}).", ""]
    per = kernels.get("per_step") or {}
    steps = kernels.get("steps")
    lines = [f"Kernel time, measured ({kernels['source']}; `{kernels['trace']}`, lines "
             f"{', '.join(kernels['lines'])}): {kernels['events']} kernel events over {steps} "
             f"traced steps ({_num(per.get('kernels'))} per step); trace window "
             f"{kernels['window_ms']:.2f} ms ({_num(per.get('window_ms'), 3)} ms per step), GPU busy "
             f"{_pct(kernels['busy_fraction'])} of it (union of kernel intervals), idle "
             f"{kernels['idle_ms']:.2f} ms; summed kernel time {kernels['kernel_ms']:.2f} ms "
             f"({_num(per.get('kernel_ms'), 3)} ms per step). The traced steps block on the "
             "parameters after each step, as the timed ones do, and XLA runs them op by op unless "
             "`XLA_FLAGS` has `--xla_enable_command_buffers_during_profiling=true` (by default "
             "profiling turns command buffers off), so the idle share bounds the untraced step's "
             f"launch gaps from above. Every kernel: `{kernels['csv']}`.",
             "", f"Top {len(kernels['top'])} kernels by summed time:", "",
             "| kernel | count | total ms | share |", "| --- | --- | --- | --- |",
             *[f"| {_cell(name)} | {count} | {ms:.3f} | {_pct(share)} |"
               for name, count, ms, share in kernels["top"]], "",
             "By family (trailing instance numbers stripped):", "",
             "| family | count | total ms | share |", "| --- | --- | --- | --- |",
             *[f"| {_cell(name)} | {count} | {ms:.3f} | {_pct(share)} |"
               for name, count, ms, share in kernels["families"]], ""]
    if kernels.get("modules"):
        lines += ["By HLO module: " + "; ".join(f"`{name}` {count} kernels, {ms:.2f} ms "
                                                f"({_pct(share)})"
                                                for name, count, ms, share in kernels["modules"])
                  + ".", ""]
    return lines


def render_report(profile: dict) -> str:
    """REPORT.md from a ``collect`` result: the four-row table, the on-the-fly overhead, the
    targets, where the time goes (measured busy shares and ``diagnosis``, else "pending the
    per-kernel table"), the full seed, the JAX step details with the measured kernel tables and the
    static HLO op counts, and the PyTorch profiler split."""
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
             "Padding: share of padded node rows, environment rows, AEV-list slots and GAT-list "
             "slots (the masked share of each layer's `(ncap, kb, heads, hidden)` gather).", "",
             "| loop | s / epoch | min / seed (200 epochs) | GPU util | loader wait | node padding "
             "| env padding | AEV-slot padding | GAT-slot padding |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, row in table:
        row = row or {}
        padding = row.get("padding", {})
        lines.append(f"| {name} | {_num(row.get('seconds_per_epoch'))} | "
                     f"{_num(row.get('minutes_per_seed'))} | {_pct(row.get('gpu_util'))} | "
                     f"{_pct(row.get('wait_fraction'))} | {_pct(padding.get('node'))} | "
                     f"{_pct(padding.get('env'))} | {_pct(padding.get('aev'))} | "
                     f"{_pct(padding.get('gat'))} |")
    overhead = _overhead(jax_rows)
    lines += ["", f"on-the-fly overhead: {_pct(overhead)}"
              + (f" (mean step {1e3 * jax_rows['onthefly']['step_seconds']['mean']:.2f} ms on the "
                 f"fly vs {1e3 * jax_rows['precomputed']['step_seconds']['mean']:.2f} ms "
                 "precomputed, same batches)" if overhead is not None else ""), "",
              "The precomputed A/B measures compute only: the environment arrays are still "
              "assembled and transferred in both modes, and the precomputed batches also carry "
              "the 352 AEV columns.", "", "## Targets", "", *_targets(profile, overhead), ""]

    busy = [f"{LABELS[mode]}: GPU busy {_pct(kernels.get('busy_fraction'))} of the traced window, "
            f"{_num((kernels.get('per_step') or {}).get('kernels'))} kernels per step"
            for mode in ("onthefly", "precomputed")
            if (kernels := (jax_rows.get(mode) or {}).get("kernels")) and not kernels.get("error")]
    lines += ["## Where the time goes", "",
              "Measured: the per-kernel tables of each JAX row below (time by kernel, family and "
              "HLO module from the perfetto trace, and the GPU busy share of the traced window); "
              "the HLO op counts are static and carry no times."
              + (f" {'; '.join(busy)}." if busy else " No kernel table in this profile."), "",
              f"Conclusion: {profile.get('diagnosis') or PENDING}.", ""]

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
        fill = (row.get("padding") or {}).get("gat_fill")
        lines += [f"## {LABELS[mode]}", "",
                  f"Step (`loss_and_grad` + `apply`, `block_until_ready`): mean "
                  f"{1e3 * step['mean']:.2f} ms, p50 {1e3 * step['p50']:.2f} ms, p95 "
                  f"{1e3 * step['p95']:.2f} ms over {row['steps']} steps after "
                  f"{row.get('warmup_steps')} warm-up steps (first step "
                  f"{_num(row.get('first_step_seconds'))} s with compilation); timed segment "
                  f"{row['segment_seconds']:.2f} s; validation pass {row['valid_seconds']:.2f} s; "
                  f"{row['steps_per_epoch']} steps per epoch; capacities ncap {cap.get('ncap')}, "
                  f"mcap {cap.get('mcap')}, kb {cap.get('kb')}, ka {cap.get('ka')}; GAT slot fill "
                  f"{_pct(fill)} of kb on the real rows; trace of "
                  f"{row.get('trace_steps')} steps in `{trace.get('dir')}` (xplane.pb and "
                  "perfetto_trace.json.gz"
                  + (f"; XLA_FLAGS `{trace['xla_flags']}`" if trace.get("xla_flags") else "")
                  + ")" + (f" FAILED: {trace['error']}" if trace.get("error") else "") + ".", "",
                  *kernel_lines(row.get("kernels")),
                  f"HLO op counts (static fallback, no kernel times): "
                  f"{row['op_summary']['source']}.", "", "| op | value |", "| --- | --- |",
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
    parser.add_argument("--trace-summary", type=Path, metavar="TRACE_DIR",
                        help="summarise an existing trace into <out>/kernels.{csv,json} (with "
                             "--trace-steps for the per-step numbers) and stop")
    args = parser.parse_args(argv)
    if args.trace_summary is not None and (args.store_root or args.seed_run):
        parser.error("--trace-summary takes --out and --trace-steps only")
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


def print_kernels(summary: dict) -> None:
    print("\n".join(kernel_lines(summary)), flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.trace_summary is not None:
        summary = trace_kernels(args.trace_summary, args.out, args.trace_steps or None)
        _write_json(args.out / KERNELS_JSON, summary)
        print_kernels(summary)
        return 0 if not summary.get("error") else 1
    if args.store_root is not None or args.seed_run is not None:
        row = profile_step(args) if args.store_root is not None else None
        profile = _read_json(args.out / PROFILE_JSON)  # after the measurement: rows of other runs
        if row is not None:
            profile.setdefault("jax", {})[args.aev] = row
            print_kernels(row.get("kernels"))
        if args.seed_run is not None:
            profile["seed"] = seed_row(args.seed_run, args.seed_window)
        _write_json(args.out / PROFILE_JSON, profile)
    print(f"wrote {write_report(args.out)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
