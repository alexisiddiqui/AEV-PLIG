# Shared variables for the Isambard jobs. Source this file; it only exports variables.
export S=/scratch/u6xq/hussain.u6xq
export TREE=$S/aevplig-jax
export JAX_PY=$S/jax-Ka/.venv/bin/python
export AEVPLIG_PY=/projects/u6xq/hussain/opig-harness/envs/aevplig/bin/python
export PARENT=$S/aevplig/work
export PAIRED=$S/aevplig-paired/work
export SBATCH_FLAGS="--partition=workq --gres=gpu:1 --cpus-per-task=72 --mem-per-cpu=2G --exclude=nid[010029,010033,010037,010038,010040,010063,010068,010078,010086,010148,010229,010241,010311,010639]"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=$TREE/work/cache
export PYTHONPATH=$TREE/src
