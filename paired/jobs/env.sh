# Environment for aevplig-paired jobs. Source it from an sbatch script (before `set -u`):
#   source /scratch/u6xq/hussain.u6xq/aevplig-paired/jobs/env.sh
export HARNESS_HOME=/projects/u6xq/hussain/opig-harness
export AEVPLIG_REPO=$HARNESS_HOME/src/AEV-PLIG-3ccac4b
export PARENT_WORK=/scratch/u6xq/hussain.u6xq/aevplig/work   # read-only inputs
export PAIRED_ROOT=/scratch/u6xq/hussain.u6xq/aevplig-paired
export WORK=$PAIRED_ROOT/work
export SRC=$PAIRED_ROOT/src
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

if [ -f "$HARNESS_HOME/envs/aevplig/bin/activate" ]; then
    source "$HARNESS_HOME/envs/aevplig/bin/activate"
else
    export PATH="$HARNESS_HOME/envs/aevplig/bin:$PATH"
fi

mkdir -p "$WORK"
cd "$WORK"
