# Source me in every AEV-PLIG job: venv, paths, and cd into the run directory.
# The AEV-PLIG scripts use CWD-relative paths (data/..., output/...), so they run from $WORK as
#   python $REPO/<script>.py ...      (the script's own dir, $REPO, becomes sys.path[0] for its local imports)
# No `module load cuda`: torch 2.6.0+cu126 brings its own CUDA runtime (only a torch-scatter source build would
# need cuda/12.6, and AEV-PLIG does not import torch-scatter).
export HARNESS_HOME=/projects/u6xq/hussain/opig-harness
export AEVPLIG_ENV=$HARNESS_HOME/envs/aevplig
export REPO=$HARNESS_HOME/src/AEV-PLIG-3ccac4b          # oxpig/AEV-PLIG 3ccac4b84d6fc1fb372cb623e705837fbf280f15
export AEVPLIG_REPO=$REPO                              # read by jobs/train_seed.py and jobs/ensemble_eval.py
export AEVPLIG_ROOT=/scratch/u6xq/hussain.u6xq/aevplig
export WORK=$AEVPLIG_ROOT/work
export JOBS=$AEVPLIG_ROOT/jobs
export AEVPLIG_LOGS=$AEVPLIG_ROOT/logs
export DATASET=pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark
# node-local scratch for many-file work (extraction, temp); callers create and remove it
export LOCAL_ROOT=/local/user/$(id -u)/aevplig
# no .pyc in the shared clone or env; unbuffered logs
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# shellcheck source=/dev/null
source "$AEVPLIG_ENV/bin/activate"
cd "$WORK"
