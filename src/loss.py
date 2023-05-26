import math
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch
import torch.nn as nn

class Label(Enum):
    POSITIVE = 1
    UNLABELED = 0

class GRABLoss(nn.Module):
    def __init__(self, loss=(lambda x: torch.sigmoid(x))) -> None:
        super().__init__()
        self.loss_func = loss
        self.positive = Label.POSITIVE
        self.unlabelled = Label.UNLABELED

    def forward(self, inp, target, b):
        positive, unlabelled = target == self.positive.value, target == self.unlabelled.value
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


class PULoss(nn.Module):
    def __init__(self, prior=torch.tensor(0.5), loss=(lambda x: torch.sigmoid(-x)), beta=0, nnpu=True) -> None:
        super().__init__()
        # check if prior is a tensor
        if not isinstance(prior, torch.Tensor):
            prior = torch.tensor(prior)
        self.prior = prior.cuda()
        self.beta = beta
        self.loss_func = loss
        self.nnpu = nnpu
        self.positive = 1
        self.unlabelled = 0
        self.min_count = torch.tensor(1.)

    def forward(self, inp, target):
        positive, unlabelled = target == self.positive, target == self.unlabelled
        positive, unlabelled = positive.type(torch.float), unlabelled.type(torch.float)
        n_pos, n_unlb = torch.max(self.min_count, torch.sum(positive)),\
             torch.max(self.min_count, torch.sum(unlabelled))
        
        y_pos = self.loss_func(inp) 
        # y_pos_inv = self.loss_func(inp, torch.zeros_like(target))
        # y_pos_inv = self.loss_func(inp, -torch.ones_like(target)) * positive

        y_unlabelled = self.loss_func(-inp) 
        # y_unlabelled = self.loss_func(inp, -torch.ones_like(target)) * unlabelled

        positive_risk = torch.sum(self.prior * positive / n_pos * y_pos) 
        negative_risk = torch.sum((unlabelled / n_unlb - self.prior * positive / n_pos) * y_unlabelled)

        
        if self.nnpu and negative_risk < -self.beta:
            # print("negative_risk", negative_risk)
            return positive_risk - self.beta
        else:
            return positive_risk + negative_risk
        
