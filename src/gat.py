import random
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, d1, d2, n_classes, num_node_features, type='gat', return_attention_weights=False):
        super().__init__()
        if type == 'gat':
            self.hid = 8
            self.in_head = 8
        else:
            self.hid = 64
            self.in_head = 1
        self.out_head = 1
        self.d1 = d1
        self.d2 = d2
        
        # 2개의 GAT layer를 쌓을 것이다.
        # ( 2번째 : multi-head attention 사용 )
        self.conv1 = GATConv(num_node_features, self.hid, heads=self.in_head, dropout=d1, return_attention_weights=return_attention_weights)
        self.fc = nn.Linear(self.hid*self.in_head, self.hid*self.in_head)
        self.conv2 = GATConv(self.hid*self.in_head, n_classes, concat=False,
                             heads=self.out_head, dropout=d1, return_attention_weights=return_attention_weights)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        #-------------------------------------#
        # x : (2708, 1433)
        # edge_index : (2, 10556)
        #-------------------------------------#
        x = F.dropout(x, p=self.d2, training=self.training)
        x = self.conv1(x, edge_index)
        #-------------------------------------#
        # x : (2708, 64)
        # 64 : hid 차원(8) x head 개수 (8)
        #-------------------------------------#
        x = F.elu(x)
        x = F.dropout(x, p=self.d2, training=self.training)
        x = self.conv2(x, edge_index)
        #-------------------------------------#
        # x : (2708, 7)
        #-------------------------------------#
        return x
