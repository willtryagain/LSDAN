import random

import torch
from torch import nn
import torch.optim as optim

from sklearn.metrics import f1_score 

import numpy as np

from GAT import GAT
from utils import *
from data_loading import parse_data, make_binary
from loss import PULoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "num_of_layers": 2,  # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
    "num_heads_per_layer": [8, 1],
    "num_features_per_layer": [CORA_NUM_INPUT_FEATURES, 64, 1],
    "add_skip_connection": True,  # hurts perf on Cora
    "bias": True,  # result is not so sensitive to bias
    "dropout": 0.9,  # result is sensitive to dropout
    "layer_type": LayerType.IMP3,  # fastest implementation enabled by default
    "num_of_epochs": 500,
    "p": 0.05
}


criteria = nn.BCELoss(reduction='mean')

# for epoch in range(config['num_of_epochs']):
dataset = 'cora'
x_, edge_index_, y_ = parse_data(dataset, False)




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
    indices, y_binary_train, y_binary_test = make_binary(y_, 3, config['p'])
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
    nnpu = False
    verbose = False
    while True:
        cur_loss = train(epoch, False, nnpu)
        if epoch == config["num_of_epochs"]: break
        
        prev_loss = cur_loss
        epoch += 1

    f1_scores.append(test(verbose, nnpu))
    print(f1_scores[-1])

print(config["p"], np.mean(f1_scores),'±' ,np.std(f1_scores))