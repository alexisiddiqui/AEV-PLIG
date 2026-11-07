#!/bin/bash
# Verify the record stores against the original pickled graphs: per record the atom count, the 15 RDKit
# columns and the bond lists (exact) and the JAX radial AEV against the pickled 352 AEV columns
# (max abs < 1e-4), on the GPU through the path training uses.
#   trainside  $PARENT/data/{pdbbind,bindingnet,bindingdb}.pickle   --require-all-pickle-ids
#   fep        $PAIRED/data/fep_benchmark_graphs.pickle
#   pgdh       $PAIRED/data/pgdh_graphs.pickle
# Each run writes $TREE/work/store/<store>/verification.json (replaced if present) and logs to
# $LOGS/aj-verify-<job>-<store>.log. All three run even if one fails; the job exits non-zero if any fails.
#
#   sbatch --dependency=afterok:<aj-build job id> /scratch/u6xq/hussain.u6xq/aevplig-jax/jobs/verify_store.sbatch
#SBATCH --job-name=aj-verify
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
#SBATCH --exclude=nid[010029,010033,010037,010038,010040,010063,010068,010078,010086,010148,010229,010241,010311,010639]
#SBATCH --output=/scratch/u6xq/hussain.u6xq/aevplig-jax/work/store/logs/%x-%j.out
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/aevplig-jax/jobs/env.sh
mkdir -p "$TMPDIR"; ulimit -c 0
export PYTHONPATH=$TREE/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
STORE=$TREE/work/store
LOGS=$STORE/logs
JID=${SLURM_JOB_ID:-$$}
TRAIN_PICKLES=("$PARENT"/data/{pdbbind,bindingnet,bindingdb}.pickle)
FEP_PICKLE=$PAIRED/data/fep_benchmark_graphs.pickle
PGDH_PICKLE=$PAIRED/data/pgdh_graphs.pickle

t0=$(date +%s)
stamp() { echo "[verify] $(date +%H:%M:%S) +$(( $(date +%s) - t0 ))s $*"; }

# 0. preflight: packages and device, stores, pickles
"$JAX_PY" -c "import jax, scipy, zstandard; print('[verify] jax', jax.__version__, jax.devices())" || {
  echo "$JAX_PY cannot import jax, scipy and zstandard" >&2; exit 1; }
for s in trainside fep pgdh; do
  [ -f "$STORE/$s/index.npz" ] || { echo "missing store $STORE/$s" >&2; exit 1; }
done
for f in "${TRAIN_PICKLES[@]}" "$FEP_PICKLE" "$PGDH_PICKLE"; do
  [ -f "$f" ] || { echo "missing pickle $f" >&2; exit 1; }
done
mkdir -p "$LOGS"
stamp "job $JID on $(hostname); stores $STORE"

run_verify() {  # run_verify <store> <args...>: one verify_store run, log in $LOGS
  local name=$1 st rc=0; shift; st=$(date +%s)
  local log=$LOGS/aj-verify-$JID-$name.log
  "$JAX_PY" -m aevplig_jax.verify_store --store "$STORE/$name" --out "$STORE/$name/verification.json" \
    "$@" > "$log" 2>&1 || rc=$?
  echo "[verify] TIMING verify_$name $(( $(date +%s) - st ))s rc=$rc (log $log)"
  tail -n 3 "$log"
  return $rc
}

fail=0
pickle_args=()
for f in "${TRAIN_PICKLES[@]}"; do pickle_args+=(--pickle "$f"); done
run_verify trainside --require-all-pickle-ids "${pickle_args[@]}" || fail=1
run_verify fep --pickle "$FEP_PICKLE" || fail=1
run_verify pgdh --pickle "$PGDH_PICKLE" || fail=1

# summary of the three reports
"$JAX_PY" - "$STORE" <<'PY'
import json, sys
from pathlib import Path
store = Path(sys.argv[1])
for name in ("trainside", "fep", "pgdh"):
    path = store / name / "verification.json"
    if not path.exists():
        print(f"VERIFY {name}: no report")
        continue
    r = json.loads(path.read_text())
    print(f"VERIFY {name}: passed {r['passed']}, counts {r['counts']}, worst {r['worst']}, "
          f"failures by source {r['failures_by_source']}, by check {r['failures_by_check']}, "
          f"capacities {r['capacities']}, {r['seconds']} s on {r['device']}")
    for failure in r["failures"]:
        print(f"  FAILED {failure['id']} ({failure['source']}): {failure['checks']} aev {failure['aev_max_abs']}")
PY
stamp "done; total $(( $(date +%s) - t0 ))s"
[ $fail = 0 ] || { echo "[verify] a verification failed (logs: $LOGS/aj-verify-$JID-*.log)" >&2; exit 1; }
