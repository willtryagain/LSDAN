import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

from loss import PULoss
from models.gcn import GCN
from data_loading import parse_planetoid_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch LSDAN')
    parser.add_argument('--dataset', default='Cora', type=str)
    parser.add_argument('--method', default='LSDAN', type=str)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--known_prior', action='store_true')

    args = parser.parse_args()

    num = torch.randint(0, 100, (1,)).item()
    torch.manual_seed(num)

    if args.dataset == "WikiCS":
        dataset = Planetoid(root='../data', name=args.dataset)
    else:
        dataset = Planetoid(root='../data', name=args.dataset)
    data, prior = parse_planetoid_data(dataset, known_prior=args.known_prior)

    if args.method[:3] == "GCN":
        model = GCN(data.num_features, 1)
    else:
        raise NotImplementedError
    model = model.to('cuda')

    if args.method[4:] == "CE":
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.method[4:] == "URE":
        loss_fn = PULoss(prior=prior, nnpu=False)
    elif args.method[4:] == "NRE":
        loss_fn = PULoss(prior=prior, nnpu=True)
    else:
        raise NotImplementedError
    optimizer = torch.optim.Adam(model.parameters())
    prev_loss = float('inf')
    num_epochs = 2000 if dataset.name == "WikiCS" else 1000

    stats = {
        "train_loss": [],
        "f1": [],
        "acc": []
    }

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out.squeeze(1), data.y_train)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        if train_loss > prev_loss:
            pass
        else:
            prev_loss = train_loss
            torch.save(model.state_dict(), "models/{}_{}.pt".format(args.dataset, args.method))
            if args.verbose:
                print("Saved model for epoch: {}".format(epoch))

        model.eval()
        out = model(data)
        pred = torch.where(out < 0.5, torch.tensor(0, device='cuda'), 
                    torch.tensor(1, device='cuda'))
        f1 = f1_score(data.y.cpu(), pred.cpu())
        acc = accuracy_score(data.y.cpu(), pred.cpu())
        stats["train_loss"].append(train_loss)
        stats["f1"].append(f1)
        stats["acc"].append(acc)

        if args.verbose:
            print("Epoch: {}, Loss: {}, F1: {}, Acc: {}".format(epoch, loss.item(), f1, acc))

    model.eval()
    out = model(data)
    pred = torch.where(out < 0.5, torch.tensor(0, device='cuda'), 
                    torch.tensor(1, device='cuda'))
    acc = accuracy_score(data.y.cpu(), pred.cpu())
    f1 = f1_score(data.y.cpu(), pred.cpu())
    file_name = "logs/{}_{}.txt".format(args.dataset, args.method)
    


    if args.verbose: print("F1: {}, Acc: {}".format(f1, acc))

    with open(file_name, 'a+') as f:
        f.write("{} {}\n".format(f1, acc))


    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(stats["train_loss"])
    axs[0].set_title("Loss")
    axs[1].plot(stats["f1"])
    axs[1].set_title("F1")
    axs[2].plot(stats["acc"])
    axs[2].set_title("Acc")
    plt.savefig("plots/{}_{}.png".format(args.dataset, args.method))
    
