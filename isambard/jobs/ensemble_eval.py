"""Test-split ensemble evaluation for AEV-PLIG, the way training.py scores its ensemble at the end: predict the
test .pt with each model, inverse-transform with that ensemble's y-scaler, average the members, report Pearson
and RMSE (plus Spearman and Kendall). Scores the retrained ensemble and the released one on the same test set.

Run from the work dir (cd $WORK):
  python $JOBS/ensemble_eval.py [--dataset pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark]
      [--tag retrain] [--released model_GATv2Net_ligsim90_fep_benchmark] [--n_models 10]
Reads   data/processed/<dataset>_test.pt (plus _train.pt only if no scaler pickle exists)
        output/trained_models/<tag>_model_GATv2Net_<dataset>_{i}.model + <tag>_model_GATv2Net_<dataset>.pickle
        output/trained_models/<released>_{i}.model + <released>.pickle
Writes  output/predictions/ensemble_eval_<dataset>_test.csv (truth, every member, both ensemble means)
Missing retrained members are skipped (reported), so it can run while the array is still going.
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.environ.get("AEVPLIG_REPO", "/projects/u6xq/hussain/opig-harness/src/AEV-PLIG-3ccac4b"))
import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch_geometric.loader import DataLoader

from helpers import model_dict, pearson, rmse
from utils import GraphDataset


def predict_scaled(model, device, loader):
    model.eval()
    out = []
    with torch.no_grad():
        for data in loader:
            out.append(model(data.to(device)).cpu())
    return torch.cat(out).numpy().reshape(-1, 1)


def metrics(t, p):
    return dict(pearson=pearson(t, p), rmse=rmse(t, p), spearman=stats.spearmanr(t, p)[0],
                kendall=stats.kendalltau(t, p)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark")
    ap.add_argument("--model", default="GATv2Net")
    ap.add_argument("--tag", default="retrain")
    ap.add_argument("--released", default="model_GATv2Net_ligsim90_fep_benchmark")
    ap.add_argument("--n_models", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--head", type=int, default=3)
    ap.add_argument("--activation_function", default="leaky_relu")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    md = os.path.join("output", "trained_models")

    retrain_prefix = f"{args.tag}_model_{args.model}_{args.dataset}"
    ensembles = {"retrained": retrain_prefix, "released": args.released}
    scalers = {}
    for name, prefix in ensembles.items():
        p = os.path.join(md, prefix + ".pickle")
        if os.path.exists(p):
            with open(p, "rb") as f:
                scalers[name] = pickle.load(f)
    if "retrained" not in scalers:  # same scaler training.py would fit: StandardScaler on the train labels
        scalers["retrained"] = GraphDataset(root="data", dataset=args.dataset + "_train").y_scaler
        print("no retrain scaler pickle; fitted the scaler on", args.dataset + "_train.pt")

    # labels in pK units: scale with any scaler, invert with the same one
    test = GraphDataset(root="data", dataset=f"{args.dataset}_{args.split}", y_scaler=scalers["retrained"])
    loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)
    y = torch.cat([d.y.view(-1, 1).cpu() for d in loader]).numpy()  # as training.py's predict() collects labels
    truth = scalers["retrained"].inverse_transform(y.reshape(-1, 1)).ravel()
    print(f"{args.split} set: {len(test)} complexes, {test.num_node_features} node features; device {device}")

    df = pd.DataFrame({"truth": truth})
    summary = []
    for name, prefix in ensembles.items():
        cols = []
        for i in range(args.n_models):
            path = os.path.join(md, f"{prefix}_{i}.model")
            if not os.path.exists(path):
                print(f"{name}: missing {path}, skipped")
                continue
            model = model_dict[args.model](node_feature_dim=test.num_node_features,
                                           edge_feature_dim=test.num_edge_features, config=args).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            df[f"{name}_{i}"] = scalers[name].inverse_transform(predict_scaled(model, device, loader)).ravel()
            m = metrics(truth, df[f"{name}_{i}"].values)
            print(f"{name} member {i}: PC {m['pearson']:.4f} RMSE {m['rmse']:.4f}")
            cols.append(f"{name}_{i}")
        if not cols:
            continue
        df[f"{name}_mean"] = df[cols].mean(axis=1)
        m = metrics(truth, df[f"{name}_mean"].values)
        summary.append(dict(ensemble=name, members=len(cols), **m))
    os.makedirs(os.path.join("output", "predictions"), exist_ok=True)
    out = os.path.join("output", "predictions", f"ensemble_eval_{args.dataset}_{args.split}.csv")
    df.to_csv(out, index=False)
    print()
    print(pd.DataFrame(summary).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("wrote", out)


if __name__ == "__main__":
    main()
