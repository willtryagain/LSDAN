import random
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

import matplotlib.pyplot as plt

from sklearn.metrics import f1_score 

from loss import PULoss
from data_loading import parse_data, make_binary

parser = argparse.ArgumentParser(description='PyTorch LSDAN')
parser.add_argument('--d1', default=0.5, type=float)
parser.add_argument('--d2', default=0.5, type=float)
parser.add_argument('--dataset', default='cora', type=str)
parser.add_argument('--p', default=0.05, type=float)
parser.add_argument('--nnpu', action='store_true')


class Data:
    def __init__(self, x, y_train, y_test, edge_index) -> None:
        self.x = x.float()
        self.edge_index = edge_index.int()
        self.indices = indices
        self.y_train = y_train.float()
        self.y_test = y_test.float()
        
    def to(self, device):
        self.x = self.x.to(device)
        self.y_train = self.y_train.to(device)
        self.y_test = self.y_test.to(device)
        self.indices = self.indices.to(device)
        self.edge_index = self.edge_index.to(device)
        return self

args = parser.parse_args()
num_classes = 1
if args.dataset == "citeseer":
    label_index = 2
elif args.dataset == "cora":
    label_index = 3


x_, edge_index_, y_ = parse_data(args.dataset)
num_node_features = x_.shape[1]
indices, y_binary_train, y_binary_test = make_binary(y_, label_index, args.p)


class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1
        
        # 2개의 GAT layer를 쌓을 것이다.
        # ( 2번째 : multi-head attention 사용 )
        self.conv1 = GATConv(num_node_features, self.hid, heads=self.in_head, dropout=args.d1)
        self.conv2 = GATConv(self.hid*self.in_head, num_classes, concat=False,
                             heads=self.out_head, dropout=args.d1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        #-------------------------------------#
        # x : (2708, 1433)
        # edge_index : (2, 10556)
        #-------------------------------------#
        x = F.dropout(x, p=args.d2, training=self.training)
        x = self.conv1(x, edge_index)
        #-------------------------------------#
        # x : (2708, 64)
        # 64 : hid 차원(8) x head 개수 (8)
        #-------------------------------------#
        x = F.elu(x)
        x = F.dropout(x, p=args.d2, training=self.training)
        x = self.conv2(x, edge_index)
        #-------------------------------------#
        # x : (2708, 7)
        #-------------------------------------#
        return torch.sigmoid(x)        
        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


f1_scores = []  

N_ITER = 10

seeds = np.random.randint(1000, size=N_ITER) 


for i in range(N_ITER):
    seed_value = seeds[i]
    np.random.seed(seed_value)
    random.seed(None)
    torch.manual_seed(seed_value)

    model = GAT().to(device)
    data = Data(x_, y_binary_train, y_binary_test, edge_index_).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)  

    model.train()
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        # criterion = nn.BCELoss()
        criterion = PULoss(nnpu=args.nnpu, beta=0.1/2)
        loss, flag = criterion(out[data.indices].squeeze(1), data.y_train[data.indices])
        # if flag: print(loss.item())
        
        loss.backward()
        optimizer.step()

    model.eval()
    output = model(data)

    pred = torch.where(output < 0.5, torch.tensor(0, device=device), 
                torch.tensor(1, device=device))

    f1 = f1_score(data.y_test[data.indices].cpu(), pred[data.indices].cpu())
    f1_scores.append(f1)
    # print()

print(np.mean(f1_scores), np.std(f1_scores))