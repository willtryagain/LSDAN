import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GATConv

import matplotlib.pyplot as plt

from sklearn.metrics import f1_score 

from loss import PULoss
from data_loading import parse_data, make_binary, Data
from gat import GAT

parser = argparse.ArgumentParser(description='PyTorch LSDAN')
parser.add_argument('--d1', default=0.55, type=float)
parser.add_argument('--d2', default=0.55, type=float)
parser.add_argument('--lambda_', default=1e-2, type=float)
parser.add_argument('--dataset', default='cora', type=str)
parser.add_argument('--p', default=0.05, type=float)
parser.add_argument('--nnpu', action='store_true')
parser.add_argument('--type', default='gat', type=str)
parser.add_argument('--seeds', default=10, type=int)




args = parser.parse_args()
label_index = -1
if args.dataset == "citeseer":
    label_index = 2
elif args.dataset == "cora":
    label_index = 3

x_, edge_index_, y_ = parse_data(args.dataset)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

overall = []
for i in range(10):
    N_SEEDS = args.seeds
    seeds_numpy = np.random.randint(0, 10000, N_SEEDS)
    f1_scores = []  


    for i in range(N_SEEDS):
        np.random.seed(seeds_numpy[i])
        torch.manual_seed(42)

        num_node_features = x_.shape[1]
        indices, y_binary_train, y_binary_test = make_binary(y_, label_index, args.p)
        model = GAT(args.d1, args.d2, 1, num_node_features, args.type).to(device)
        data = Data(x_, y_binary_train, y_binary_test, edge_index_, indices).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=args.lambda_)  

        model.train()
        for epoch in range(500):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            criterion = PULoss(nnpu=args.nnpu, beta=0)
            loss, flag = criterion(out[data.indices].squeeze(1), data.y_train[data.indices])
            loss.backward()
            optimizer.step()

        model.eval()
        output = model(data)

        pred = torch.where(output < 0.5, torch.tensor(0, device=device), 
                    torch.tensor(1, device=device))

        f1 = f1_score(data.y_test[data.indices].cpu(), pred[data.indices].cpu())
        f1_scores.append(f1)

    overall.append(np.mean(f1_scores))
    print(f"Mean F1 score: {np.mean(f1_scores)}")

print(np.mean(overall), np.std(overall))