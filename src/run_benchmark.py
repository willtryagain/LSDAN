import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
# from torch_geometric.nn.models import GCN
from torch_geometric.datasets import Planetoid

import argparse

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt

from loss import PULoss
from models.gcn import GCN


# add command line arguments
parser = argparse.ArgumentParser(description='PyTorch LSDAN')
# dataset 
parser.add_argument('--dataset', default='Cora', type=str)
# nnpu or not
parser.add_argument('--nnpu', action='store_true')
# verbose
parser.add_argument('--verbose', action='store_true')


known_table = {}

known_table["GCN+NRE"] = {
    "CiteSeer": {"f1": "66.2 +/- 1.1", "acc": "93.2 +/- 0.2"},
    "Cora": {"f1": "76.7 +/- 0.9", "acc": "92.7 +/- 0.2"}

}

known_table["GCN+URE"] = {
    "CiteSeer": {"f1": "42.6 +/- 1.7", "acc": "90.9 +/- 0.2"},
     "Cora": {"f1": "50.9 +/- 0.8", "acc": "88.0 +/- 0.1"}
}

def parse_planetoid_data(dataset, seed):

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
    data = data.to('cuda')
    data.y_train = data.y_train.to('cuda')
    data.y = data.y.to('cuda')
    data.train_mask = data.train_mask.to('cuda')
    data.test_mask = data.test_mask.to('cuda')
    data.edge_index = data.edge_index.to('cuda')

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
    return data, prior

if __name__ == '__main__':

    args = parser.parse_args()

    num = torch.randint(0, 100, (1,)).item()
    torch.manual_seed(num)
    dataset = Planetoid(root='../data', name=args.dataset)
    data, prior = parse_planetoid_data(dataset, num)
    

    num_classes = 1
    num_layers = 2
    model = GCN(data.num_features, num_classes)
    model = model.to('cuda')
    loss_fn = PULoss(prior=prior, nnpu=args.nnpu)
    optimizer = torch.optim.Adam(model.parameters())
    model.train()
    best_train_loss = float('inf')
    patience = 1000  # number of consecutive epochs to wait before stopping
    counter = 0   # counter to keep track of consecutive epochs with increased loss

    # store loss, f1, and accuracy
    train_losses = []
    f1s = []
    accs = []

    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out.squeeze(1), data.y_train)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        if train_loss > best_train_loss:
            counter += 1
            if counter == patience:
                break
        
        else:
            best_train_loss = train_loss
            counter = 0
            torch.save(model.state_dict(), "models/{}_nnpu_{}.pt".format(args.dataset, args.nnpu))
            if args.verbose:
                print("Saved model for epoch: {}".format(epoch))

        model.eval()
        out = model(data)
        pred = torch.where(out < 0.5, torch.tensor(0, device='cuda'), 
                    torch.tensor(1, device='cuda'))
        f1 = f1_score(data.y.cpu(), pred.cpu())
        acc = accuracy_score(data.y.cpu(), pred.cpu())
        train_losses.append(train_loss)

        f1s.append(f1)
        accs.append(acc)



        if args.verbose:
            print("Epoch: {}, Loss: {}, F1: {}, Acc: {}".format(epoch, loss.item(), f1, acc))
        
        method = "GCN+{}".format("NRE" if args.nnpu else "URE")
        known_f1 =  known_table[method][args.dataset]["f1"]
        known_f1 = float(known_f1.split(" ")[0])/100
        # if f1 >= known_f1:
        #     print("method: {}, dataset: {}, epoch: {}".format(method, args.dataset, epoch))
        #     break

    # load the best model
    model.load_state_dict(torch.load("models/{}_nnpu_{}.pt".format(args.dataset, args.nnpu)))
    
    model.eval()
    out = model(data)
    pred = torch.where(out < 0.5, torch.tensor(0, device='cuda'), 
                    torch.tensor(1, device='cuda'))
    acc = accuracy_score(data.y.cpu(), pred.cpu())
    f1 = f1_score(data.y.cpu(), pred.cpu())
    file_name = "logs/{}_nnpu_{}.txt".format(args.dataset, args.nnpu)
    
    # check if file exists

    mode = 'a+'
    if not os.path.exists(file_name):
        mode = 'w'
    else:
        with open(file_name, 'r') as f:
            if len(f.readlines()) == 10:
                mode = 'w'

  

    # if f1 < 0.5:
    #     print("num: {}".format(num))
    if args.verbose: print("F1: {}, Acc: {}".format(f1, acc))

    with open(file_name, mode) as f:
        f.write("{} {}\n".format(f1, acc))


    # 3 subplots
    # 1. loss
    # 2. f1
    # 3. acc
    # 

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(train_losses)
    axs[0].set_title("Loss")
    axs[1].plot(f1s)
    axs[1].set_title("F1")
    axs[2].plot(accs)
    axs[2].set_title("Acc")
    plt.savefig("plots/{}_nnpu_{}.png".format(args.dataset, args.nnpu))
    
