"""Featurise fep_benchmark.csv with the clone's process_and_predict (no prediction).

Runs process_data(config) and generate_graphs(config) from $AEVPLIG_REPO/process_and_predict.py,
then builds data/processed/fep_benchmark.pt exactly as make_predictions does
(GraphDatasetPredict with graph_id = row order of data/fep_benchmark_processed.csv).
The clone uses CWD-relative paths (data/PDB_Atom_Keys.csv, data/<name>_graphs.pickle,
data/processed/), so this chdirs into --work-dir first. process_data derives its output name with
dataset_csv.split(".")[0], which breaks on absolute paths containing a dot (hussain.u6xq), so the
input CSV is always passed as the relative path data/fep_benchmark.csv (copied there if needed).
Outputs (under --work-dir): data/fep_benchmark_graphs.pickle, data/fep_benchmark_processed.csv,
data/processed/fep_benchmark.pt, data/fep_featurise_summary.json.
"""
import os
import sys

sys.path.insert(0, os.environ["AEVPLIG_REPO"])

import argparse
import filecmp
import json
import pickle
import shutil
import types

DATA_NAME = "fep_benchmark"
REL_CSV = os.path.join("data", DATA_NAME + ".csv")
REL_PROCESSED = os.path.join("data", DATA_NAME + "_processed.csv")
REL_GRAPHS = os.path.join("data", DATA_NAME + "_graphs.pickle")
REL_PT = os.path.join("data", "processed", DATA_NAME + ".pt")
REL_KEYS = os.path.join("data", "PDB_Atom_Keys.csv")


def stage_inputs(csv_path, work_dir):
    """Put the CSV at <work>/data/fep_benchmark.csv and the clone's atom keys at <work>/data/."""
    os.makedirs(os.path.join(work_dir, "data", "processed"), exist_ok=True)
    dst = os.path.join(work_dir, REL_CSV)
    if os.path.realpath(csv_path) != os.path.realpath(dst):
        shutil.copyfile(csv_path, dst)
    keys_src = os.path.join(os.environ["AEVPLIG_REPO"], REL_KEYS)
    keys_dst = os.path.join(work_dir, REL_KEYS)
    if not os.path.isfile(keys_dst) or not filecmp.cmp(keys_src, keys_dst, shallow=False):
        shutil.copyfile(keys_src, keys_dst)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", required=True, help="fep_benchmark.csv from fep_prepare.py")
    ap.add_argument("--work-dir", required=True, help="directory holding data/ (the clone's CWD layout)")
    ap.add_argument("--num-workers", type=int, default=72)
    ap.add_argument("--skip-validation", action="store_true", help="skip the Biopandas PDB check")
    args = ap.parse_args(argv)

    csv_path = os.path.abspath(args.csv)
    work_dir = os.path.abspath(args.work_dir)
    stage_inputs(csv_path, work_dir)
    os.chdir(work_dir)

    os.environ["OMP_NUM_THREADS"] = str(args.num_workers)
    os.environ["MKL_NUM_THREADS"] = str(args.num_workers)
    import pandas as pd
    import torch
    import process_and_predict as pap  # the clone's module (torchani, qcelemental, torchani_mod)
    from utils import GraphDatasetPredict

    torch.set_num_threads(args.num_workers)
    config = types.SimpleNamespace(dataset_csv=REL_CSV, data_name=DATA_NAME, num_workers=args.num_workers,
                                   skip_validation=args.skip_validation, device=torch.device("cpu"))
    assert config.dataset_csv.split(".")[0] + "_processed.csv" == REL_PROCESSED

    n_input = len(pd.read_csv(REL_CSV))
    pap.process_data(config)
    pap.generate_graphs(config)

    # .pt exactly as make_predictions builds it, minus the prediction
    data = pd.read_csv(REL_PROCESSED)
    with open(REL_GRAPHS, "rb") as fh:
        graphs = pickle.load(fh)
    dropped_process = sorted(set(pd.read_csv(REL_CSV)["unique_id"]) - set(data["unique_id"]))
    no_graph = [u for u in data["unique_id"] if u not in graphs]
    if no_graph:  # make_predictions would KeyError; drop them so graph_id stays the processed-CSV row
        print(f"WARNING: {len(no_graph)} rows have no graph; removing them from {REL_PROCESSED}: {no_graph}")
        data = data[data["unique_id"].isin(graphs)].reset_index(drop=True)
        data.to_csv(REL_PROCESSED, index=False)
    data["graph_id"] = range(len(data))
    if os.path.exists(REL_PT):
        os.remove(REL_PT)
    dataset = GraphDatasetPredict(root="data", dataset=DATA_NAME, ids=list(data["unique_id"]),
                                  graph_ids=list(data["graph_id"]), graphs_dict=graphs)

    summary = {"n_input": n_input, "n_processed": len(data), "n_graphs": len(graphs),
               "n_pt": len(dataset), "num_node_features": int(dataset.num_node_features),
               "num_edge_features": int(dataset.num_edge_features),
               "dropped_by_process_data": dropped_process, "dropped_no_graph": no_graph}
    with open(os.path.join("data", "fep_featurise_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"input {n_input}  processed {len(data)}  graphs {len(graphs)}  pt {len(dataset)}  "
          f"node features {dataset.num_node_features}  edge features {dataset.num_edge_features}")
    print(f"dropped by process_data ({len(dropped_process)}): {dropped_process}")
    if dataset.num_node_features != 367 or dataset.num_edge_features != 4:
        sys.exit("ERROR: expected 367 node and 4 edge features")


if __name__ == "__main__":
    main()
