# PATCHED COPY of AEV-PLIG training.py (oxpig/AEV-PLIG 3ccac4b) for one-seed-per-job training on Isambard.
# Every change is marked "# PATCH". Changes: (1) the clone's dir goes on sys.path (this file lives in jobs/, not
# in the clone); (2) --seed_index picks ONE of the 10 seeds (-1 = all ten, the original loop); (3) --tag fixes the
# output names to output/trained_models/<tag>_model_<model>_<dataset>_<i>.model and the y-scaler to
# output/trained_models/<tag>_model_<model>_<dataset>.pickle (original: <timestamp>_... and the scaler in the CWD);
# (4) the scaler is written before training (atomic rename; it is identical for every seed) so ensemble_eval.py
# can use it while other seeds still run; (5) per-seed test predictions go to output/predictions/.
# Run from the work dir:  python jobs/train_seed.py --seed_index=3 --tag=retrain --epochs=200 ...
import os, sys  # PATCH (1)
sys.path.insert(0, os.environ.get("AEVPLIG_REPO", "/projects/u6xq/hussain/opig-harness/src/AEV-PLIG-3ccac4b"))  # PATCH (1)
import torch
import random
import time
import torch.nn as nn
from torch_geometric.loader import DataLoader
from helpers import rmse, pearson, model_dict
from utils import GraphDataset, init_weights
import os
import pandas as pd
import argparse
import numpy as np
import pickle

def predict(model, device, loader, y_scaler=None):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)

    return y_scaler.inverse_transform(total_labels.numpy().flatten().reshape(-1,1)).flatten(), y_scaler.inverse_transform(total_preds.detach().numpy().flatten().reshape(-1,1)).flatten()



def train(model, device, train_loader, optimizer, epoch, loss_fn):
    log_interval = 100
    model.train()
    total_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).to(device))
        loss.backward()
        optimizer.step()
        total_loss += (loss.item()*len(data.y))
        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]'.format(epoch,
                                                             batch_idx * len(data.y),
                                                             len(train_loader.dataset),
                                                             100. * batch_idx / len(train_loader)))
    
    print("Loss for epoch {}: {:.4f}".format(epoch, total_loss/len(train_loader.dataset)))
    return total_loss/len(train_loader.dataset)



def _train(model, device, loss_fn, train_loader, valid_loader, optimizer, n_epochs, y_scaler, model_output_dir, model_file_name):
    best_pc = -1.1
    pcs = []
    for epoch in range(n_epochs):
    
        _ = train(model, device, train_loader, optimizer, epoch + 1, loss_fn)
        
        G, P = predict(model, device, valid_loader, y_scaler)

        current_pc = pearson(G, P)
        pcs.append(current_pc)
        
        low = np.maximum(epoch-7,0)
        avg_pc = np.mean(pcs[low:epoch+1])
        if(avg_pc > best_pc):
            torch.save(model.state_dict(), os.path.join(model_output_dir, model_file_name))
            best_pc = avg_pc  
            
        print('The current validation set Pearson correlation:', current_pc)
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GATv2Net')
    parser.add_argument('--dataset', type=str, default='pdbbind_U_bindingnet_ligsim90')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--head', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.00012291937615434127)
    parser.add_argument('--activation_function', type=str, default='leaky_relu')
    parser.add_argument('--seed_index', type=int, default=-1, help='PATCH (2): index 0-9 into the seed list; -1 = all')
    parser.add_argument('--tag', type=str, default='retrain', help='PATCH (3): output file prefix')
    args = parser.parse_args()
    return args

def train_NN(args):
    modeling = model_dict[args.model]
    model_st = modeling.__name__
    
    batch_size = args.batch_size
    LR = args.lr
    n_epochs = args.epochs

    print('Train for {} epochs: '.format(n_epochs))

    dataset = args.dataset

    print('Running dataset {} on model {}.'.format(dataset, model_st))
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_output_dir = os.path.join("output", "trained_models")
    
    train_data = GraphDataset(root='data', dataset=dataset+'_train', y_scaler=None)
    valid_data = GraphDataset(root='data', dataset=dataset+'_valid', y_scaler=train_data.y_scaler)
    test_data = GraphDataset(root='data', dataset=dataset+'_test', y_scaler=train_data.y_scaler)

    seeds = [100, 123, 15, 257, 2, 2012, 3752, 350, 843, 621]
    # PATCH (4): scaler first, under the fixed name, via an atomic rename (several array tasks write it)
    os.makedirs(model_output_dir, exist_ok=True)
    scaler_file = os.path.join(model_output_dir, args.tag + '_model_' + model_st + '_' + dataset + '.pickle')
    tmp = scaler_file + '.tmp' + str(os.getpid())
    with open(tmp, 'wb') as f:
        pickle.dump(train_data.y_scaler, f)
    os.replace(tmp, scaler_file)
    # PATCH (2): one seed per job
    runs = list(enumerate(seeds)) if args.seed_index < 0 else [(args.seed_index, seeds[args.seed_index])]
    for i, seed in runs:
        print('Seed index {} seed {}'.format(i, seed))  # PATCH (2)
        random.seed(seed)
        torch.manual_seed(int(seed))
        
        model_file_name = args.tag + '_model_' + model_st + '_' + dataset + '_' + str(i) + '.model'  # PATCH (3)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        if(torch.cuda.is_available()):
            print("GPU is available")
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print('Device state:', device)

        model = modeling(node_feature_dim=train_data.num_node_features, edge_feature_dim=train_data.num_edge_features, config=args)
        model.apply(init_weights)
    
        print("The number of node features is ", train_data.num_node_features)
        print("The number of edge features is ", train_data.num_edge_features)
    
        weight_decay = 0
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    
        model.to(device)
        _train(model, device, loss_fn, train_loader, valid_loader, optimizer, n_epochs, train_data.y_scaler, model_output_dir, model_file_name)
        
        model.load_state_dict(torch.load(os.path.join(model_output_dir, model_file_name)))
        
        G_test, P_test = predict(model, device, test_loader, train_data.y_scaler)

        if(i == runs[0][0]):  # PATCH (2): first run, not seed index 0
            df_test = pd.DataFrame(data=G_test, index=range(len(G_test)), columns=['truth'])
        
        col = 'preds_' + str(i)
        df_test[col] = P_test
    
    df_test['preds'] = df_test.iloc[:,1:].mean(axis=1)

    # PATCH (3)/(4): scaler already written above; PATCH (5): keep this job's test predictions
    os.makedirs(os.path.join("output", "predictions"), exist_ok=True)
    suffix = 'all' if args.seed_index < 0 else str(args.seed_index)
    df_test.to_csv(os.path.join("output", "predictions", args.tag + '_' + dataset + '_test_seed' + suffix + '.csv'), index=False)
    
    test_preds = np.array(df_test['preds'])
    test_truth = np.array(df_test['truth'])
    test_ens_pc = pearson(test_truth, test_preds)
    test_ens_rmse = rmse(test_truth, test_preds)
    print("Ensemble test PC:", test_ens_pc)
    print("Ensemble test RMSE:", test_ens_rmse)



if __name__ == "__main__":
    start_time = time.time()
    
    args = parse_args()
    
    train_NN(args)
    
    print("Total time is %s seconds" % (time.time() - start_time))
    
    
