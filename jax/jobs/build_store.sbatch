#!/bin/bash
# Record stores from the raw archives: $TREE/work/store/{pdbbind,bindingnet,bindingdb,trainside,fep,pgdh}
#   pdbbind, bindingnet, bindingdb  one store per processed CSV ($PARENT/data/<source>_processed.csv),
#                                   built side by side with 24 workers each
#   trainside                       the three merged in create_pytorch_data.py order (split labels
#                                   train/valid/test/excluded)
#   fep, pgdh                       generic builds of $PAIRED/data/{fep_benchmark,pgdh}.csv (label column pK)
# The archives stay in $S/aevplig/raw (read-only). Extraction, the BindingNet template merge and the four
# data/... links are copied from isambard/jobs/featurise.sbatch, but the links live on node-local disk
# under $X/root (passed as --root), so only the stores and logs land on scratch. The extraction is
# removed at exit (trap, also on failure). Existing stores are not overwritten: remove them first.
#
#   mkdir -p /scratch/u6xq/hussain.u6xq/aevplig-jax/work/store/logs
#   sbatch /scratch/u6xq/hussain.u6xq/aevplig-jax/jobs/build_store.sbatch
#
# Archives (relative to $S/aevplig/raw or absolute) and their layouts:
#   PDBBIND_REFINED       PDBbind_v2020_refined.tar.gz   refined-set/<pdb>/<pdb>_{protein.pdb,ligand.mol2,...}
#   PDBBIND_GENERAL       PDBbind_v2020_other_PL.tar.gz  v2020-other-PL/<pdb>/<pdb>_{protein.pdb,ligand.mol2,...}
#   BINDINGNET            BindingNetv1.tar.gz            ./BindingNetv1/<pdb>/target_<T>/<c>/<pdb>_<T>_<c>.sdf
#   BINDINGNET_TEMPLATES  Crystal_Templates_for_BindingNet1.tar.gz  ./Crystal_Templates_for_BindingNet1/<pdb>/rec_h_opt.pdb
#   BINDINGDB             surflex.tar                    <folder>/<pdb>.pdb + <folder>/*.mol2 at the tar root
#SBATCH --job-name=aj-build
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH --exclude=nid[010029,010033,010037,010038,010040,010063,010068,010078,010086,010148,010229,010241,010311,010639]
#SBATCH --output=/scratch/u6xq/hussain.u6xq/aevplig-jax/work/store/logs/%x-%j.out
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/aevplig-jax/jobs/env.sh
mkdir -p "$TMPDIR"; ulimit -c 0
export PYTHONPATH=$TREE/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RAW=$S/aevplig/raw
STORE=$TREE/work/store
LOGS=$STORE/logs
KEYS=$PARENT/data/PDB_Atom_Keys.csv
LOCAL_ROOT=/local/user/$(id -u)/aevplig   # as in isambard/jobs/env.sh
JID=${SLURM_JOB_ID:-$$}
X=$LOCAL_ROOT/aj-build-$JID
ROOT=$X/root
: "${PDBBIND_REFINED:=PDBbind_v2020_refined.tar.gz}"
: "${PDBBIND_GENERAL:=PDBbind_v2020_other_PL.tar.gz}"
: "${BINDINGNET:=BindingNetv1.tar.gz}"
: "${BINDINGNET_TEMPLATES:=Crystal_Templates_for_BindingNet1.tar.gz}"
: "${BINDINGDB:=surflex.tar}"
LINKS=(data/pdbbind/refined-set data/pdbbind/general-set data/bindingnet/from_chembl_client data/bindingdb/surflex)
SOURCES=(pdbbind bindingnet bindingdb)
GENERIC=(fep:fep_benchmark:64 pgdh:pgdh:8)   # store:csv stem:workers

t0=$(date +%s)
stamp() { echo "[build] $(date +%H:%M:%S) +$(( $(date +%s) - t0 ))s $*"; }
declare -A T=()   # phase -> seconds

# 0. preflight: package, inputs, no stale stores
"$AEVPLIG_PY" -c "import zstandard" 2>/dev/null || {
  echo "zstandard is not installed in $AEVPLIG_PY; install it with: uv pip install --python $AEVPLIG_PY zstandard" >&2
  exit 1; }
for f in "$KEYS" "$PARENT"/data/{pdbbind,bindingnet,bindingdb}_processed.csv "$PAIRED"/data/{fep_benchmark,pgdh}.csv; do
  [ -f "$f" ] || { echo "missing input $f" >&2; exit 1; }
done
for s in "${SOURCES[@]}" trainside fep pgdh; do
  if [ -e "$STORE/$s" ]; then echo "$STORE/$s exists; remove it before rebuilding" >&2; exit 1; fi
done
mkdir -p "$LOGS"

cleanup() {
  local rc=$?
  local c0; c0=$(date +%s)
  rm -rf "$X"
  rmdir "$LOCAL_ROOT" 2>/dev/null || true
  echo "[build] cleanup: removed $X in $(( $(date +%s) - c0 ))s; exit code $rc"
  echo "[build] /local check: $(ls -A "$LOCAL_ROOT" 2>/dev/null | wc -l) entries left in $LOCAL_ROOT"
}
trap cleanup EXIT
stamp "job $JID on $(hostname); stores $STORE; extraction $X"

abspath() { case $1 in /*) echo "$1" ;; *) echo "$RAW/$1" ;; esac; }
extract() {  # extract <archive> <dest>
  local a=$1 d=$2 s; s=$(date +%s)
  mkdir -p "$d"
  case $a in
    *.tar.gz|*.tgz) tar -I pigz -xf "$a" -C "$d" ;;
    *.tar)          tar -xf "$a" -C "$d" ;;
    *) echo "unknown archive type $a" >&2; return 1 ;;
  esac
  echo "[build] extracted $(basename "$a") in $(( $(date +%s) - s ))s"
}

# 1. extract the five archives in parallel
p0=$(date +%s); xpids=()
for pair in "PDBBIND_REFINED pdbbind_refined" "PDBBIND_GENERAL pdbbind_general" "BINDINGNET bindingnet" \
            "BINDINGNET_TEMPLATES bindingnet" "BINDINGDB surflex"; do
  set -- $pair; a=$(abspath "${!1}")
  [ -f "$a" ] || { echo "$1: no file $a" >&2; exit 1; }
  stamp "extract $1 $(basename "$a") ($(du -h "$a" | cut -f1)) -> $X/$2"
  extract "$a" "$X/$2" & xpids+=($!)
done
for p in "${xpids[@]}"; do wait "$p" || { echo "extraction failed" >&2; exit 1; }; done
T[extract]=$(( $(date +%s) - p0 ))
stamp "extraction done in ${T[extract]}s: $(du -sh "$X" | cut -f1)"

# 2. BindingNet merge: give every BindingNetv1/<pdb>/ its crystal template receptor
BN=$X/bindingnet/BindingNetv1
CT=$X/bindingnet/Crystal_Templates_for_BindingNet1
[ -d "$BN" ] && [ -d "$CT" ] || { echo "missing $BN or $CT" >&2; ls "$X/bindingnet" >&2; exit 1; }
nl=0; nm=0
for d in "$BN"/*/; do
  d=${d%/}; p=${d##*/}
  if [ -f "$CT/$p/rec_h_opt.pdb" ]; then ln -sfn "$CT/$p/rec_h_opt.pdb" "$d/rec_h_opt.pdb"; nl=$((nl + 1))
  else nm=$((nm + 1)); fi
done
stamp "BindingNet merge: $nl pdb dirs linked to a template rec_h_opt.pdb, $nm without a template"

# 3. the data/... layout of the generate scripts, under $ROOT (node-local)
declare -A SRC=(
  [data/pdbbind/refined-set]=$X/pdbbind_refined/refined-set
  [data/pdbbind/general-set]=$X/pdbbind_general/v2020-other-PL
  [data/bindingnet/from_chembl_client]=$BN
  [data/bindingdb/surflex]=$X/surflex
)
for l in "${LINKS[@]}"; do
  [ -d "${SRC[$l]}" ] || { echo "missing ${SRC[$l]}" >&2; exit 1; }
  mkdir -p "$ROOT/$(dirname "$l")"
  ln -sfn "${SRC[$l]}" "$ROOT/$l"
  stamp "$l -> ${SRC[$l]} ($(ls "${SRC[$l]}" | wc -l) entries)"
done

# 4. the three training-side sources side by side, 24 workers each
run_build() {  # run_build <name> <args...>: one build_store run, log in $LOGS
  local name=$1 st; shift; st=$(date +%s)
  local log=$LOGS/aj-build-$JID-$name.log
  if "$AEVPLIG_PY" -m aevplig_jax.build_store "$@" --keys "$KEYS" --out "$STORE/$name" > "$log" 2>&1; then
    echo "[build] TIMING build_$name $(( $(date +%s) - st ))s ok"
  else
    local rc=$?; echo "[build] TIMING build_$name $(( $(date +%s) - st ))s FAILED rc=$rc (log $log)"; tail -n 20 "$log"; return $rc
  fi
}
p0=$(date +%s); declare -A BP=()
for s in "${SOURCES[@]}"; do
  stamp "build $s"
  run_build "$s" --source "$s" --processed-csv "$PARENT/data/${s}_processed.csv" --root "$ROOT" --workers 24 &
  BP[$s]=$!
done
fail=0
for s in "${SOURCES[@]}"; do wait "${BP[$s]}" || fail=1; done
T[build_sources]=$(( $(date +%s) - p0 ))
[ $fail = 0 ] || { echo "a source build failed (logs: $LOGS/aj-build-$JID-*.log)" >&2; exit 1; }
stamp "sources built in ${T[build_sources]}s"

# 5. trainside = pdbbind + bindingnet + bindingdb, in create_pytorch_data.py order
p0=$(date +%s)
"$AEVPLIG_PY" - "$STORE" <<'PY'
import sys
from pathlib import Path
from aevplig_jax.store import merge_stores
store = Path(sys.argv[1])
merge_stores([store / name for name in ("pdbbind", "bindingnet", "bindingdb")], store / "trainside")
PY
T[merge]=$(( $(date +%s) - p0 ))
stamp "trainside merged in ${T[merge]}s"

# 6. generic stores (absolute structure paths in the CSVs), side by side
p0=$(date +%s); declare -A GP=()
for spec in "${GENERIC[@]}"; do
  IFS=: read -r name stem workers <<< "$spec"
  stamp "build $name from $PAIRED/data/$stem.csv"
  run_build "$name" --generic --csv "$PAIRED/data/$stem.csv" --label-column pK --workers "$workers" &
  GP[$name]=$!
done
fail=0
for name in "${!GP[@]}"; do wait "${GP[$name]}" || fail=1; done
T[build_generic]=$(( $(date +%s) - p0 ))
[ $fail = 0 ] || { echo "a generic build failed (logs: $LOGS/aj-build-$JID-*.log)" >&2; exit 1; }
stamp "generic stores built in ${T[build_generic]}s"

# 7. summary
"$AEVPLIG_PY" - "$STORE" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
from aevplig_jax.store import Store
store = Path(sys.argv[1])
for name in ("pdbbind", "bindingnet", "bindingdb", "fep", "pgdh"):
    s = json.loads((store / name / "build_summary.json").read_text())
    print(f"COUNT {name}: {s['records']} records from {s['rows']} rows, {len(s['failures'])} failures, "
          f"splits {s['splits']}, {s['seconds']} s")
    for failure in s["failures"][:20]:
        print(f"  FAILED {failure['id']}: {failure['reason']}")
for name in ("trainside", "fep", "pgdh"):
    with Store(store / name) as s:
        splits = Counter(s.records[i]["split"] for i in s.ids)
        print(f"COUNT {name} store: {len(s)} records, splits {dict(splits)}, "
              f"max dims [L, E, P, A, KA] = {s.dims.max(axis=0).tolist() if len(s) else []}, "
              f"records.bin {(store / name / 'records.bin').stat().st_size / 1e6:.1f} MB")
PY
du -sh "$STORE"/*/
for k in "${!T[@]}"; do echo "[build] TIMING $k ${T[$k]}s"; done
stamp "done; total $(( $(date +%s) - t0 ))s"
