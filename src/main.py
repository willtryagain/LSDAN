import random
import argparse

import torch
from torch import nn
import torch.optim as optim

from sklearn.metrics import f1_score 

import numpy as np

from data_loading import parse_data, make_binary
from loss import PULoss


parser = argparse.ArgumentParser(description='PyTorch LSDAN')
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--dataset', default='cora', type=str)
parser.add_argument('--p', default=0.05, type=float)
parser.add_argument('--bias', action='store_true')
parser.add_argument('--nnpu', action='store_true')
parser.add_argument('--add_skip_connection', action='store_true')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

config = {
    "num_of_layers": 2,  # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
    "num_heads_per_layer": [8, 1],
    "add_skip_connection": args.bias,  # hurts perf on Cora
    "bias": args.add_skip_connection,  # result is not so sensitive to bias
    "dropout": args.dropout,  # result is sensitive to dropout
    "num_of_epochs": 500,
    "p": args.p,
}

if args.dataset == "citeseer":
    config['class_label'] = 2
    config["num_features_per_layer"] = [3703, 64, 1]
elif args.dataset == "cora":
    config['class_label'] = 3
    config["num_features_per_layer"] = [1433, 64, 1]

dataset = args.dataset
x_, edge_index_, y_ = parse_data(dataset)


f1_scores = []  

N_ITER = 10

seeds = np.random.randint(1000, size=N_ITER) 


for i in range(N_ITER):
    np.random.seed(seeds[i])
    random.seed(seeds[i])
    torch.manual_seed(seeds[i])

    model = GAT(
        num_of_layers=config['num_of_layers'],
        num_heads_per_layer=config['num_heads_per_layer'],
        num_features_per_layer=config['num_features_per_layer'],
        add_skip_connection=config['add_skip_connection'],
        bias=config['bias'],
        dropout=config['dropout'],
        layer_type=config['layer_type'],
        log_attention_weights=False  # no need to store attentions, used only in playground.py for visualizations
    ).to(device)

    def train(epoch, verbose=False, nnpu=False):
        model.train()
        optimizer.zero_grad()
        output = model(graph_data)[0].index_select(node_dim, all_indices)
        # if (torch.max(output).item() > 0): print(torch.max(output).item())
        output = torch.sigmoid(output)
        criterion = PULoss(nnpu=nnpu)
        loss = criterion(output[indices].view(-1), y_binary_train[indices]) # !CHECK

        loss.backward()
        optimizer.step()
        if verbose:
            print('Epoch: {}\tLoss:{}'.format(epoch, loss.item()))
        return loss.item()

    def test(verbose=True, nnpu=False):
        model.eval()
        output = model(graph_data)[0].index_select(node_dim, all_indices)
        output = torch.sigmoid(output)
        criterion = PULoss(nnpu=nnpu)
        loss = criterion(output[indices].view(-1), y_binary_test[indices])
        pred = torch.where(output < 0.5, torch.tensor(0, device=device), 
            torch.tensor(1, device=device))
        f1 = f1_score(y_binary_test[indices].cpu(), pred[indices].cpu())
        if verbose:
            print('f1:{}\tLoss:{}'.format(f1, loss.item()))
        return f1

    node_dim = 0
    indices, y_binary_train, y_binary_test = make_binary(y_, config['class_label'], config['p'])
    x = x_.cuda().float()
    edge_index = edge_index_.cuda().long()
    y = torch.from_numpy(y_).cuda().float()
    all_indices = torch.arange(x.shape[0]).cuda()
    indices = indices.cuda()
    y_binary_train = y_binary_train.cuda().float()
    y_binary_test = y_binary_test.cuda().float()
    graph_data = (x, edge_index)
    device = torch.device("cuda")

    optimizer =  optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = config["num_of_epochs"]
    epoch = 0
    nnpu = args.nnpu
    verbose = False
    while True:
        cur_loss = train(epoch, verbose, nnpu)
        if epoch == config["num_of_epochs"]: break
        prev_loss = cur_loss
        epoch += 1

    f1_scores.append(test(verbose, nnpu))

print(config["p"], np.mean(f1_scores),'±' ,np.std(f1_scores))