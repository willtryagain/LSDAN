import pickle
import zipfile
import json


import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import scipy.sparse as sp
import torch
from torch.hub import download_url_to_file
from torch.utils.data import DataLoader, Dataset

from utils import *

import numpy as np
from scipy import sparse
import torch as T
import torch


class Data:
    def __init__(self, x, y_train, y_test, edge_index, indices) -> None:
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

def normalize(mx):
    """mat gets row-normalized"""
    rowsum = np.array(mx.sum(1)).astype(float)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def get_sparse_tensor(mat):
	mat = mat.tocoo().astype(np.float32)
	indices = T.from_numpy(
		np.vstack((mat.row, mat.col)).astype(np.int64)
	)
	values = T.from_numpy(mat.data)
	return T.sparse.FloatTensor(indices, values, T.Size(mat.shape))

def parse_data(dataset):
    x = []
    with open('../data/{}/{}.txt'.format(dataset, 'feature'), 'r') as f:
        lines = f.readlines()
    for line in lines:
        row = line.strip().split('\t')
        row = [int(item) for item in row]
        x.append(row)
    x = normalize(np.array(x))
    assert np.max(x) <= 1
    assert np.min(x) >= 0
    x = T.from_numpy(x)

    y = []
    with open('../data/{}/{}.txt'.format(dataset, 'group'), 'r') as f:
        lines = f.readlines()
    for line in lines:
        row = line.strip().split('\t')
        row = int(row[1])
        y.append(row)
    y = np.array(y)
    assert np.min(y) >= 0

    E = []
    with open('../data/{}/{}.txt'.format(dataset, 'graph'), 'r') as f:
        lines = f.readlines()

    seen_edges = set()
    source_ids, target_ids = [], []
    for line in lines:
        row = line.strip().split('\t')
        u = int(row[0])
        v = int(row[1])
        if (u, v) not in seen_edges:
            seen_edges.add((u, v))
            source_ids.append(u)
            target_ids.append(v)

    edge_index = torch.from_numpy(np.row_stack((source_ids, target_ids)))

    if dataset == 'citeseer':
        assert x.shape[0] == 3312
        assert len(lines) == 4732
        assert y.max() + 1 == 6
        assert x.shape[1] == 3703

    elif dataset == 'cora':
        assert x.shape[0] == 2708
        assert len(lines) == 5429
        assert y.max() + 1 == 7
        assert x.shape[1] == 1433
    
    return x, edge_index, y

def parse_data_lsdan(dataset, k=4):
    x = []
    with open('../data/{}/{}.txt'.format(dataset, 'feature'), 'r') as f:
        lines = f.readlines()
    for line in lines:
        row = line.strip().split('\t')
        row = [int(item) for item in row]
        x.append(row)
    x = normalize(np.array(x))
    assert np.max(x) <= 1
    assert np.min(x) >= 0
    x = T.from_numpy(x)

    y = []
    with open('../data/{}/{}.txt'.format(dataset, 'group'), 'r') as f:
        lines = f.readlines()
    for line in lines:
        row = line.strip().split('\t')
        row = int(row[1])
        y.append(row)
    y = np.array(y)
    assert np.min(y) >= 0

    E = []
    A = np.zeros((x.shape[0], x.shape[0]))
    with open('../data/{}/{}.txt'.format(dataset, 'graph'), 'r') as f:
        lines = f.readlines()

    source_ids, target_ids = [], []
    for line in lines:
        row = line.strip().split('\t')
        u = int(row[0])

        v = int(row[1])
        A[u, v] = 1
    A_cur = A.copy()

    for i in range(k):
        if i != 0: A_cur = A_cur @ A
        cur_source_ids, cur_target_ids = [], []
        cnt = 0
        for u in range(A_cur.shape[0]):
            for v in range(A_cur.shape[1]):
                if A_cur[u, v] == 0: continue
                cur_source_ids.append(u)
                cur_target_ids.append(v)
                cnt += 1
        source_ids.append(cur_source_ids)
        target_ids.append(cur_target_ids)

    edge_indices = []
    for i in range(k):
        edge_index = torch.from_numpy(np.row_stack((source_ids[i], target_ids[i])))
        edge_indices.append(edge_index)


    if dataset == 'citeseer':
        assert x.shape[0] == 3312
        assert len(lines) == 4732
        assert y.max() + 1 == 6
        assert x.shape[1] == 3703

    elif dataset == 'cora':
        assert x.shape[0] == 2708
        assert len(lines) == 5429
        assert y.max() + 1 == 7
        assert x.shape[1] == 1433
    
    return x, edge_indices, y

def make_binary(y, class_label, p):
    mask = y == class_label
    y_binary_test = mask.astype(int)
    y_binary_train = np.zeros_like(y_binary_test)
    P = np.nonzero(mask)[0]
    N = np.nonzero(~mask)[0]
    k = len(P)
    N_equal = np.random.choice(N, k, False)
    indices = np.concatenate((P, N_equal))
    P_train = np.random.choice(P, int(k * p), False)
    y_binary_train[P_train] = 1

    np.random.shuffle(indices)
    indices = torch.from_numpy(indices)
    y_binary_train = torch.from_numpy(y_binary_train)
    y_binary_test = torch.from_numpy(y_binary_test)

    return indices, y_binary_train, y_binary_test


def parse_planetoid_data(dataset, known_prior=False, device=torch.device('cpu')):

    data = dataset[0]
    class_freq = data.y.bincount().float()
    max_class = class_freq.argmax()
    data.y = torch.where(data.y == max_class, torch.tensor(1), torch.tensor(0))
    data.y = data.y.float()


    pos_idx = torch.where(data.y == 1)[0]
    neg_idx = torch.where(data.y == 0)[0]
    # pos_idx = pos_idx[torch.randperm(pos_idx.size(0))]
    # neg_idx = neg_idx[torch.randperm(neg_idx.size(0))]


    # sample 50% of the positive class
    P = pos_idx[:int(pos_idx.size(0)/2)]
    P_t = pos_idx[int(pos_idx.size(0)/2):]
    # concatenate the positive and negative indices
    U = torch.cat([P_t, neg_idx])
    data.U_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.U_mask[U] = 1
    # shuffle the indices
    # U = U[torch.randperm(U.size(0))]
    # set traininging adn testing indices to be the whole dataset
    data.train_mask = torch.ones(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.ones(data.num_nodes, dtype=torch.bool)

    # contruct the training labels 
    data.y_train = torch.zeros(data.num_nodes)
    data.y_train[P] = 1

    prior = torch.tensor(P_t.size(0) / (P_t.size(0) + neg_idx.size(0)))

    # to cuda
    data = data.to(device)
    data.y_train = data.y_train.to(device)
    data.y = data.y.to(device)
    data.train_mask = data.train_mask.to(device)
    data.test_mask = data.test_mask.to(device)
    data.edge_index = data.edge_index.to(device)

    if dataset.name == "Cora":
        assert data.num_features == 1433
        assert data.num_nodes == 2708
        assert data.num_edges == 5278 * 2
        assert data.y.sum() == 818
    
    elif dataset.name == "CiteSeer":
        assert data.num_features == 3703
        assert data.num_nodes == 3327
        assert data.num_edges == 4552 * 2
        assert data.y.sum() == 701

    elif dataset.name == "WikiCS":
        assert data.num_features == 300
        assert data.num_nodes == 11701  
        assert data.num_edges == 215603 * 2
        assert data.y.sum() == 2679

    if not known_prior:
        prior = torch.tensor(P.size(0) / data.num_nodes)
    return data, prior



if __name__ == "__main__":
    x, edge_indices, y =  parse_data('cora')
    # x, edge_indices, y =  parse_data('citeseer')

    for i in range(4):
        print(edge_indices[i][0].shape)