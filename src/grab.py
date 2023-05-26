import torch
from torch_geometric.datasets import Planetoid

from loss import GRABLoss
from models.gcn import GCN
from lbp import LBP
from data_loading import parse_planetoid_data


class GRAB:
    def __init__(self, data, epochs=1000) -> None:
        self.data = data
        self.model = GCN(data.num_features, 1)
        self.epochs = epochs

    def train(self, B):
        optimizer = torch.optim.Adam(self.model.parameters())
        loss_func = GRABLoss()

        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(self.data)
            loss = loss_func(out, self.data.y_train.view(-1, 1), B)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print('Epoch: {:03d}, Loss: {:.5f}'.format(epoch, loss.item()))

        self.model.eval()
        out = self.model(self.data)
        l_new = loss_func(out, self.data.y.view(-1, 1), B)
        pred = out > 0.5
        prior_new = torch.sum(pred[self.data.U_mask]) / torch.sum(self.data.U_mask)

        return prior_new, l_new
    
    def run(self):
        l_new = torch.tensor(float('inf'))
        prior_new = torch.tensor(0.0)

        epoch = 0
        while True:
            l, prior = l_new, prior_new
            lbp = LBP(prior, data.edge_index, data.U_mask)
            lbp.run()
            B = lbp.belief()
            prior_new, l_new = self.train(B)
            epoch += 1
            print(epoch, prior_new, l_new)
            if l < l_new:
                break

if __name__ == "__main__":
    dataset = Planetoid(root='../data', name="Cora")
    data, prior = parse_planetoid_data(dataset, known_prior=True)
    grab = GRAB(data)
    grab.run()
