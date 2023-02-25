import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GATConv



class GAT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1
        
        # 2개의 GAT layer를 쌓을 것이다.
        # ( 2번째 : multi-head attention 사용 )
        self.conv1 = GATConv(num_node_features, self.hid, heads=self.in_head, dropout=args.d1)
        self.fc = nn.Linear(self.hid*self.in_head, self.hid*self.in_head)
        self.conv2 = GATConv(self.hid*self.in_head, num_classes, concat=False,
                             heads=self.out_head, dropout=args.d1)

    def forward(self, data):
        x, edge_indices = data.x, data.edge_indices
        #-------------------------------------#
        # x : (2708, 1433)
        # edge_index : (2, 10556)
        #-------------------------------------#
        x = F.dropout(x, p=args.d2, training=self.training)
        H = [self.conv1(x, edge_index[i]) for _ in range(4)]
        x = 
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
