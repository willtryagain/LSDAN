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

def normalize(mat):
	"""mat gets row-normalized"""
	row_sum = np.array(mat.sum(1)) + 1e-5
	reciprocal = np.reciprocal(row_sum).flatten()
	reciprocal[np.isinf(reciprocal)] = 0
	reciprocal_mat = sparse.diags(reciprocal)
	return reciprocal_mat.dot(mat)

def get_sparse_tensor(mat):
	mat = mat.tocoo().astype(np.float32)
	indices = T.from_numpy(
		np.vstack((mat.row, mat.col)).astype(np.int64)
	)
	values = T.from_numpy(mat.data)
	return T.sparse.FloatTensor(indices, values, T.Size(mat.shape))

def parse_data(dataset, verbose=True):

    x = []
    with open('../data/{}/{}.txt'.format(dataset, 'feature'), 'r') as f:
        lines = f.readlines()
    for line in lines:
        row = line.strip().split('\t')
        row = [int(item) for item in row]
        x.append(row)
    x = T.from_numpy(normalize(np.array(x)))

    y = []
    with open('../data/{}/{}.txt'.format(dataset, 'group'), 'r') as f:
        lines = f.readlines()
    for line in lines:
        row = line.strip().split('\t')
        row = int(row[1])
        y.append(row)
    y = np.array(y)

    E = []
    with open('../data/{}/{}.txt'.format(dataset, 'graph'), 'r') as f:
        lines = f.readlines()

    seen_edges = set()
    source_ids, target_ids = []
    for line in lines:
        row = line.strip().split('\t')
        u = int(row[0])
        v = int(row[1])
        if (u, v) not in seen_edges:
            seen_edges.add((u, v))
            source_ids.append(u)
            target_ids.append(v)

    edge_index = np.row_stack((source_ids, target_ids))

    # E = np.array(E)
    # A = sparse.coo_matrix(
    #     (np.ones(E.shape[0]), (E[:, 0], E[:, 1])),
    #     (len(y), len(y)),
    #     np.float32
    # )
    # A += A.T.multiply(A.T > A) - A.multiply(A < A.T) #? logic
    # A = normalize(A + sparse.eye(len(y)))
    # A = get_sparse_tensor(A)




    if verbose:
        print("#Edges:", len(lines))
        print("#Classes:", y.max() + 1)
    
    return x, edge_index, y
