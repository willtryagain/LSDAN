import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class PULoss(nn.Module):
    def __init__(self, prior=torch.tensor(0.5), loss=nn.SoftMarginLoss(reduction='none'), beta=0, nnpu=True) -> None:
        super().__init__()
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
        
        y_pos = self.loss_func(inp, torch.ones_like(target)) * positive
        # y_pos_inv = self.loss_func(inp, torch.zeros_like(target)) * positive
        y_pos_inv = self.loss_func(inp, -torch.ones_like(target)) * positive

        # y_unlabelled = self.loss_func(inp, torch.zeros_like(target)) * unlabelled
        y_unlabelled = self.loss_func(inp, -torch.ones_like(target)) * unlabelled

        positive_risk = self.prior * torch.sum(y_pos) / n_pos
        negative_risk = torch.sum(y_unlabelled) / n_unlb - self.prior * torch.sum(y_pos_inv) / n_pos

        if self.nnpu and negative_risk < -self.beta:
            return positive_risk
        else:
            return positive_risk + negative_risk