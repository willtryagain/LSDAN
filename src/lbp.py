from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from data_loading import parse_planetoid_data



class Label(Enum):
    POSITIVE = 1
    UNLABELLED = 0

class LBP:
    def __init__(self, prior, edges, U_mask, homophily=0.9) -> None:
        self.prior = prior
        self.edges = edges
        self.mask = U_mask
        self.homophily = homophily
        self.num_nodes = self.mask.shape[0]
        self.node_potential = torch.ones(self.num_nodes, 2)
        self.node_potential[~self.mask][Label.POSITIVE.value] = 1
        self.node_potential[~self.mask][Label.UNLABELLED.value] = 0
        self.node_potential[self.mask][Label.POSITIVE.value] = prior
        self.node_potential[self.mask][Label.UNLABELLED.value] = 1 - prior
        self.messages = torch.fill_(torch.empty(self.edges.shape[1], 2), 0.5)
        self.edge_to_index = {}
        self.neighbors = {}
        for index in range(self.edges.shape[1]):
            i, j = self.edges[0][index], self.edges[1][index]
            i, j = int(i), int(j)
            if j not in self.neighbors:
                self.neighbors[j] = []
            self.neighbors[j].append(i)
            self.edge_to_index[(i, j)] = index
        
    def get_index_from_edge(self, u, v):
        return self.edge_to_index[(u, v)]

    def edge_potential(self, u, v):
        if u == v:
            return self.homophily
        else:
            return 1 - self.homophily
        
    def convergence(self, prev_messages):
        diff = torch.max(torch.abs(self.messages - prev_messages))
        return diff < 1e-4
        
    def run(self):
        epoch = 0
        while True:
            epoch += 1
            prev_messages = self.messages.clone()
            # self.message_product()
            for index in range(self.edges.shape[1]):
                i, j = self.edges[0][index], self.edges[1][index]
                i, j = int(i), int(j)
                for v in [Label.POSITIVE.value, Label.UNLABELLED.value]:
                    self.messages[index][v] = 0
                    for u in [Label.POSITIVE.value, Label.UNLABELLED.value]:
                        psi = self.edge_potential(u, v)
                        cur_value = self.node_potential[i][u] * psi 
                        for k in self.neighbors[i]:
                            if k == j: continue
                            index_ki = self.get_index_from_edge(k, i)
                            cur_value *= prev_messages[index_ki][u]
                        self.messages[index][v] += cur_value

            # check if messages sum to 1
            s = torch.sum(self.messages, dim=1)
            assert torch.all(torch.abs(s - torch.ones(s.shape)) < 1e-4)

            mask = self.messages != prev_messages
            print(epoch)
            print("message", self.messages[mask])
            print("prev", prev_messages[mask])
                    

            if epoch % 10 == 0:
                print("Epoch: {}".format(epoch), "Error: {}".format(torch.max(torch.abs(self.messages - prev_messages))))
            if self.convergence(prev_messages):
                # self.message_product()
                print("LBP converged in {} epochs".format(epoch))
                break
        
    def belief(self):
        b = torch.zeros(self.num_nodes, 2)
        for j in range(self.num_nodes):
            for u in [Label.POSITIVE.value, Label.UNLABELLED.value]:
                b[j][u] = self.node_potential[j][u]
                for k in self.neighbors[j]:
                    index_kj = self.get_index_from_edge(k, j)
                    b[j][u] *= self.messages[index_kj][u]
            # print(b[j])
            if torch.sum(b[j]) != 0:
                b[j] /= torch.sum(b[j])
            else: 
                b[j] = torch.tensor([0.5, 0.5])

        return b

if __name__ == "__main__":
    dataset = Planetoid(root='../data', name="Cora")
    data, prior = parse_planetoid_data(dataset, known_prior=False)
    prior = torch.tensor(0)
    lbp = LBP(prior, data.edge_index, data.U_mask)
    lbp.run()
    b = lbp.belief()
    print(b)