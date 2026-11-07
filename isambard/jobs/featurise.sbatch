#!/bin/bash
# AEV-PLIG featurisation: raw archives -> data/{pdbbind,bindingnet,bindingdb}.pickle -> data/processed/<DATASET>_{train,valid,test}.pt
# The archives stay on scratch ($AEVPLIG_ROOT/raw). They are extracted to node-local /local, symlinked into
# work/data/ under the folder names the generate scripts expect, and the extraction and links are removed at
# exit (trap, also on failure). Only the three pickles, the three .pt files and logs land on scratch.
#
#   sbatch /scratch/u6xq/hussain.u6xq/aevplig/jobs/featurise.sbatch
#
# Archives (relative to raw/ or absolute; defaults below) and their layouts:
#   PDBBIND_REFINED       PDBbind_v2020_refined.tar.gz   refined-set/<pdb>/<pdb>_{protein.pdb,ligand.mol2,...}
#   PDBBIND_GENERAL       PDBbind_v2020_other_PL.tar.gz  v2020-other-PL/<pdb>/<pdb>_{protein.pdb,ligand.mol2,...}
#   BINDINGNET            BindingNetv1.tar.gz            ./BindingNetv1/<pdb>/target_<T>/<c>/<pdb>_<T>_<c>.sdf
#   BINDINGNET_TEMPLATES  Crystal_Templates_for_BindingNet1.tar.gz  ./Crystal_Templates_for_BindingNet1/<pdb>/rec_h_opt.pdb
#   BINDINGDB             surflex.tar                    <folder>/<pdb>.pdb + <folder>/*.mol2 at the tar root
# BindingNet merge: BindingNetv1/<pdb>/rec_h_opt.pdb -> Crystal_Templates_for_BindingNet1/<pdb>/rec_h_opt.pdb
# (symlink), then data/bindingnet/from_chembl_client -> BindingNetv1/.
#
# Optional:
#   SKIP_GENERATE=pdbbind,bindingdb   reuse the existing data/<name>.pickle instead of regenerating it
#   A jobs/generate_<name>_graphs.py, if present, is run instead of the clone's script (PYTHONPATH=$REPO).
#SBATCH --job-name=aevplig-feat
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=06:00:00
#SBATCH --exclude=nid[010029,010033,010037,010038,010040,010063,010068,010078,010086,010148,010229,010241,010311,010639]
#SBATCH --output=/scratch/u6xq/hussain.u6xq/aevplig/logs/%x-%j.out
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/aevplig/jobs/env.sh
RAW=$AEVPLIG_ROOT/raw
JID=${SLURM_JOB_ID:-$$}
X=$LOCAL_ROOT/feat-$JID
: "${PDBBIND_REFINED:=PDBbind_v2020_refined.tar.gz}"
: "${PDBBIND_GENERAL:=PDBbind_v2020_other_PL.tar.gz}"
: "${BINDINGNET:=BindingNetv1.tar.gz}"
: "${BINDINGNET_TEMPLATES:=Crystal_Templates_for_BindingNet1.tar.gz}"
: "${BINDINGDB:=surflex.tar}"
: "${SKIP_GENERATE:=}"
LINKS=(data/pdbbind/refined-set data/pdbbind/general-set data/bindingnet/from_chembl_client data/bindingdb/surflex)

cleanup() {
  local rc=$?
  for l in "${LINKS[@]}"; do [ -L "$WORK/$l" ] && rm -f "$WORK/$l"; done
  rmdir "$WORK/data/pdbbind" "$WORK/data/bindingnet" "$WORK/data/bindingdb" 2>/dev/null || true
  local c0; c0=$(date +%s)
  rm -rf "$X"
  rmdir "$LOCAL_ROOT" 2>/dev/null || true
  echo "[featurise] cleanup: removed $X in $(( $(date +%s) - c0 ))s; links removed; exit code $rc"
  echo "[featurise] /local check: $(ls -A "$LOCAL_ROOT" 2>/dev/null | wc -l) entries left in $LOCAL_ROOT"
}
trap cleanup EXIT
t0=$(date +%s)
stamp() { echo "[featurise] $(date +%H:%M:%S) +$(( $(date +%s) - t0 ))s $*"; }
declare -A T=()   # phase -> seconds
stamp "job $JID on $(hostname); work $WORK; extraction $X"

abspath() { case $1 in /*) echo "$1" ;; *) echo "$RAW/$1" ;; esac; }
for l in "${LINKS[@]}"; do
  if [ -e "$WORK/$l" ] && [ ! -L "$WORK/$l" ]; then echo "$WORK/$l exists and is not a symlink" >&2; exit 1; fi
done

extract() {  # extract <archive> <dest>
  local a=$1 d=$2 s; s=$(date +%s)
  mkdir -p "$d"
  case $a in
    *.tar.gz|*.tgz) tar -I pigz -xf "$a" -C "$d" ;;
    *.tar)          tar -xf "$a" -C "$d" ;;
    *) echo "unknown archive type $a" >&2; return 1 ;;
  esac
  echo "[featurise] extracted $(basename "$a") in $(( $(date +%s) - s ))s"
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

# 3. symlinks into work/data
declare -A SRC=(
  [data/pdbbind/refined-set]=$X/pdbbind_refined/refined-set
  [data/pdbbind/general-set]=$X/pdbbind_general/v2020-other-PL
  [data/bindingnet/from_chembl_client]=$BN
  [data/bindingdb/surflex]=$X/surflex
)
for l in "${LINKS[@]}"; do
  [ -d "${SRC[$l]}" ] || { echo "missing ${SRC[$l]}" >&2; exit 1; }
  mkdir -p "$WORK/$(dirname "$l")"
  ln -sfn "${SRC[$l]}" "$WORK/$l"
  stamp "$l -> ${SRC[$l]} ($(ls "${SRC[$l]}" | wc -l) entries)"
done
python "$JOBS/featurise_check.py" check
stamp "all input files present"

# 4. the three generate scripts are single-process: run them side by side
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 TQDM_MININTERVAL=60 PYTHONPATH=$REPO
p0=$(date +%s); declare -A GP=()
for s in pdbbind bindingnet bindingdb; do
  if [[ ",$SKIP_GENERATE," == *",$s,"* ]]; then
    [ -f "data/$s.pickle" ] || { echo "SKIP_GENERATE has $s but data/$s.pickle is missing" >&2; exit 1; }
    stamp "generate $s skipped: reusing data/$s.pickle"; continue
  fi
  script=$REPO/generate_${s}_graphs.py
  [ -f "$JOBS/generate_${s}_graphs.py" ] && script=$JOBS/generate_${s}_graphs.py
  log=$AEVPLIG_LOGS/featurise-$JID-$s.log
  stamp "generate $s: $script -> $log"
  ( st=$(date +%s)
    if python "$script" > "$log" 2>&1; then echo "[featurise] TIMING generate_$s $(( $(date +%s) - st ))s ok"
    else rc=$?; echo "[featurise] TIMING generate_$s $(( $(date +%s) - st ))s FAILED rc=$rc"; exit $rc; fi ) &
  GP[$s]=$!
done
fail=0
for s in "${!GP[@]}"; do wait "${GP[$s]}" || fail=1; done
T[generate]=$(( $(date +%s) - p0 ))
for s in "${!GP[@]}"; do echo "--- tail $s"; tail -c 2000 "$AEVPLIG_LOGS/featurise-$JID-$s.log" | tr '\r' '\n' | tail -n 6; done
[ $fail = 0 ] || { echo "a generate script failed (logs: $AEVPLIG_LOGS/featurise-$JID-*.log)" >&2; exit 1; }
stamp "generate done in ${T[generate]}s; pickles: $(ls -la data/*.pickle | awk '{print $5, $9}' | tr '\n' ' ')"

# 5. .pt files (stale ones would be loaded instead of rebuilt)
rm -f data/processed/${DATASET}_{train,valid,test}.pt
export OMP_NUM_THREADS=72 MKL_NUM_THREADS=72
p0=$(date +%s)
python "$REPO/create_pytorch_data.py"
T[create_pytorch_data]=$(( $(date +%s) - p0 ))
stamp "create_pytorch_data.py done in ${T[create_pytorch_data]}s"

# 6. summary
echo "=== sizes"
ls -la data/*.pickle data/processed/${DATASET}_*.pt
python - <<'EOF'
import os, sys, pickle
import pandas as pd
sys.path.insert(0, os.environ["REPO"])
for n in ("pdbbind", "bindingnet", "bindingdb"):
    rows = len(pd.read_csv(f"data/{n}_processed.csv", index_col=0))
    with open(f"data/{n}.pickle", "rb") as f:
        keys = len(pickle.load(f))
    print(f"COUNT {n}: {rows} rows in {n}_processed.csv, {keys} keys in {n}.pickle")
from utils import GraphDataset
ds = os.environ["DATASET"]
for s in ("train", "valid", "test"):
    print(f"COUNT {ds}_{s}.pt: {len(GraphDataset(root='data', dataset=f'{ds}_{s}'))} graphs")
EOF
for k in "${!T[@]}"; do echo "[featurise] TIMING $k ${T[$k]}s"; done
stamp "done; total $(( $(date +%s) - t0 ))s"
