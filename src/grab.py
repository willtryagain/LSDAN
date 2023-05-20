import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

Label = {
    "-1": 0,
    "1": 1
}

class GRABLoss(nn.Module):
    def __init__(self, loss=(lambda x: torch.sigmoid(x))) -> None:
        super().__init__()
        self.loss_func = loss
        self.positive = Label["1"]
        self.unlabelled = Label["-1"]

    def forward(self, inp, target, b):
        positive, unlabelled = target == self.positive, target == self.unlabelled
        positive, unlabelled = positive.type(torch.float), unlabelled.type(torch.float)
        n_pos, n_unlb = torch.sum(positive), torch.sum(unlabelled)

        # inp [n, 1] to [n, 2] where the second column is 1 - inp
        inp = torch.cat((inp, 1 - inp), dim=1)
        target = torch.cat((target, 1 - target), dim=1)
        # element wise dot product
        prod = torch.sum(inp * target, dim=-1)
        y_pos = self.loss_func(prod) * positive
        
        prod = torch.sum(inp * b, dim=-1)
        y_unlabelled = self.loss_func(prod) * unlabelled
        positive_risk = torch.sum(y_pos) / n_pos
        unlb_risk = torch.sum(y_unlabelled) / n_unlb
        
        return positive_risk + unlb_risk
    
def GRAB(data):
    l_new = float("inf")
    prior_new = 0
    epoch = 0

    while True:
        l, prior = l_new, prior_new

        if epoch % 10 == 0:
            print("GRAB epoch:", epoch)

        B = LBP(prior, data)
        print(B)

        model = GCN(data.num_features, 1)
        optimizer = torch.optim.Adam(model.parameters())
        loss_func = GRABLoss()

        for epoch in range(1000):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = loss_func(out, data.y_train.view(-1, 1), B)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print('Epoch: {:03d}, Loss: {:.5f}'.format(epoch, loss.item()))

        model.eval()
        out = model(data)
        l_new = loss_func(out, data.y.view(-1, 1), B)
        pred = out > 0.5
        prior_new = torch.sum(pred[data.PU_mask]) / torch.sum(data.PU_mask)

        if l_new > l:
            break

        l = l_new
        epoch += 1

    return prior_new, l_new, model
