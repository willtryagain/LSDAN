from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid



class Label(Enum):
    POSITIVE = 1
    UNLABELLED = 0

class LBP:
    def __init__(self, prior, edges, U_mask, homophily=0.9) -> None:
        self.prior = prior
        self.edges = edges
        self.mask = U_mask
        self.homophily = homophily
        self.node_potential = torch.ones(self.mask.shape[0], 2)
        self.node_potential[self.mask][Label.POSITIVE.value] = prior
        self.node_potential[self.mask][Label.UNLABELLED.value] = 1 - prior
        self.node_potential[~self.mask][Label.POSITIVE.value] = 1
        self.node_potential[~self.mask][Label.UNLABELLED.value] = 0
        self.num_nodes = self.mask.shape[0]
        self.messages = torch.fill_(torch.empty(self.edges.shape[1], 2), 0.5)
        self.message_prod = torch.ones(self.num_nodes, 2)
        self.edge_to_index = {}
        for index in range(self.edges.shape[1]):
            i, j = self.edges[0][index], self.edges[1][index]
            i, j = int(i), int(j)
            self.edge_to_index[(i, j)] = index
        
    def get_index_from_edge(self, u, v):
        return self.edge_to_index[(u, v)]

    def edge_potential(self, u, v):
        if u == v:
            return self.homophily
        else:
            return 1 - self.homophily
        
    def message_product(self):
        self.message_prod = torch.ones(self.num_nodes, 2)
        for index in range(self.edges.shape[1]):
            i, j = self.edges[0][index], self.edges[1][index]
            i, j = int(i), int(j)
            for u in [Label.POSITIVE.value, Label.UNLABELLED.value]:
                self.message_prod[j][u] *= self.messages[index][u]
        
    def convergence(self, prev_messages):
        diff = torch.max(torch.abs(self.messages - prev_messages))
        return diff < 1e-4
        
    def run(self):
        epoch = 0
        while True:
            epoch += 1
            prev_messages = self.messages.clone()
            self.message_product()
            for index in range(self.edges.shape[1]):
                i, j = self.edges[0][index], self.edges[1][index]
                i, j = int(i), int(j)
                for v in [Label.POSITIVE.value, Label.UNLABELLED.value]:
                    self.messages[index][v] = 0
                    for u in [Label.POSITIVE.value, Label.UNLABELLED.value]:
                        psi = self.edge_potential(u, v)
                        self.messages[index][v] += self.node_potential[i][u] * psi * self.message_prod[i][u] / prev_messages[self.get_index_from_edge(j, i)][u]
                    
            print("Epoch: {}".format(epoch), "Error: {}".format(torch.max(torch.abs(self.messages - prev_messages))))
            if self.convergence(prev_messages):
                self.message_product()
                print("Converged in {} epochs".format(epoch))
                break
        
    def belief(self):
        b = torch.zeros(self.num_nodes, 2)
        for j in range(self.num_nodes):
            b[j][Label.POSITIVE.value] = self.node_potential[j][Label.POSITIVE.value] * self.message_prod[j][Label.POSITIVE.value]
            b[j][Label.UNLABELLED.value] = self.node_potential[j][Label.UNLABELLED.value] * self.message_prod[j][Label.UNLABELLED.value]
            b[j] /= torch.sum(b[j])
        return b



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
    dataset = Planetoid(root='../data', name="Cora")
    data, prior = parse_planetoid_data(dataset, known_prior=False)
    lbp = LBP(prior, data.edge_index, data.U_mask)
    lbp.run()
    b = lbp.belief()
    print(b[data.U_mask])