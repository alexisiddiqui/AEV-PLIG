#!/bin/bash
# Build the AEV-PLIG env on Isambard-AI (aarch64, GH200): $HARNESS_HOME/envs/aevplig, a uv venv on a uv-managed
# CPython 3.11 ($HARNESS_HOME/cache/uv-python). The repo's aev-plig-linux.yml pins x86 conda builds for Python 3.8,
# so the env is built by hand: torch 2.6.0+cu126 (the harness's boltz2 version; aarch64/sbsa wheel), numpy 1.26.4,
# and the yml's pins for the pure-python packages. torch-scatter is skipped: no AEV-PLIG module imports it, and
# torch-geometric 2.5 does GATv2Conv and global pooling without it.
# Submit:  sbatch /scratch/u6xq/hussain.u6xq/aevplig/jobs/build_env.sbatch   (log: build-logs/aevplig.log)
#SBATCH --job-name=aevplig-build
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --exclude=nid[010029,010033,010037,010038,010040,010063,010068,010078,010086,010148,010229,010241,010311,010639]
#SBATCH --output=/scratch/u6xq/hussain.u6xq/aevplig/logs/%x-%j.out
set -euo pipefail
H=/projects/u6xq/hussain/opig-harness
ENV=$H/envs/aevplig
REPO=$H/src/AEV-PLIG-3ccac4b
UV=${UV:-$HOME/miniconda3/bin/uv}
exec > >(tee "$H/build-logs/aevplig.log") 2>&1
echo "[aevplig build] $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname)"

export UV_CACHE_DIR=$H/cache/uv UV_PYTHON_INSTALL_DIR=$H/cache/uv-python UV_HTTP_TIMEOUT=600
export PYTHONDONTWRITEBYTECODE=1
# build/unpack dirs on node-local disk, removed at exit
LOCAL=/local/user/$(id -u)/aevplig-build-${SLURM_JOB_ID:-$$}
mkdir -p "$LOCAL"
export TMPDIR=$LOCAL
trap 'rm -rf "$LOCAL"' EXIT

PY=$ENV/bin/python
if [ ! -f "$ENV/.complete" ]; then
  rm -rf "$ENV"
  "$UV" python install 3.11
  "$UV" venv -q --python 3.11 --prompt aevplig "$ENV"
  "$UV" pip install --python "$PY" torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
  # pure-python / wheel pins from aev-plig-linux.yml (numpy <2; setuptools <81 keeps pkg_resources, which
  # torchani_mod/__init__.py imports; lark-parser, h5py and importlib_metadata are torchani_mod imports)
  "$UV" pip install --python "$PY" \
    numpy==1.26.4 torch-geometric==2.5.3 torchani==2.2.4 rdkit==2024.3.3 qcelemental==0.28.0 biopandas==0.4.1 \
    pandas==2.0.3 scikit-learn==1.3.2 scipy==1.10.1 networkx==3.1 tqdm "setuptools<81" lark-parser \
    importlib_metadata h5py requests
fi

echo "[aevplig build] import test"
cd "$LOCAL"
"$PY" - "$REPO" <<'EOF'
import sys, warnings
sys.path.insert(0, sys.argv[1])
warnings.filterwarnings("ignore", message="cuaev not installed")
warnings.filterwarnings("ignore", message="Dependency not satisfied")
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(),
      "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
import torch_geometric, torchani, rdkit, qcelemental, biopandas, pandas, sklearn, numpy, scipy
from rdkit import Chem
from biopandas.pdb import PandasPdb
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool
from torch_geometric.loader import DataLoader
import torchani_mod
import model_defs, utils, helpers
for m in (torch_geometric, torchani, rdkit, qcelemental, biopandas, pandas, sklearn, numpy, scipy):
    print(m.__name__, getattr(m, "__version__", "?"))
print("torchani_mod", torchani_mod.__file__)
try:
    import torch_scatter
    print("torch_scatter present", torch_scatter.__version__)
except ImportError:
    print("torch_scatter absent (not needed)")
# tiny GPU forward of the AEV-PLIG GNN
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch_geometric.data import Data, Batch
class C: activation_function = "leaky_relu"; head = 3; hidden_dim = 256
net = helpers.model_dict["GATv2Net"](node_feature_dim=367, edge_feature_dim=4, config=C).to(dev).eval()
g = [Data(x=torch.randn(10, 367), edge_index=torch.randint(0, 10, (2, 20)), edge_attr=torch.randn(20, 4)) for _ in range(4)]
with torch.no_grad():
    out = net(Batch.from_data_list(g).to(dev))
print("GATv2Net forward ok", tuple(out.shape), out.device)
EOF
echo "[aevplig build] uv pip freeze"
"$UV" pip freeze --python "$PY"
touch "$ENV/.complete"
chmod -R g+rX "$ENV" "$REPO" 2>/dev/null || true
echo "[aevplig build] env files: $(find "$ENV" | wc -l)"
echo "[aevplig build] done $(date -Is)"
