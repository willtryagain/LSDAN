import torch 
from torch_geometric.nn.models import GAT
# from data_loading import parse_data, make_binary
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

root = ''
dataset = Planetoid(root, name="Cora")



# dataset = 'cora'


# config = {
#     "num_of_layers": 2,  # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
#     "num_heads_per_layer": [8, 1],
#     "num_of_epochs": 500,
#     'p': 0.03
# }

# if dataset == "citeseer":
#     config['class_label'] = 2
#     config["num_features_per_layer"] = [3703, 64, 1]
# elif dataset == "cora":
#     config['class_label'] = 3
#     config["num_features_per_layer"] = [1433, 64, 1]

# x_, edge_index_, y_ = parse_data(dataset)
# x = x_.cuda().float()
# indices, y_binary_train, y_binary_test = make_binary(y_, config['class_label'], config['p'])


print(dataset)
model = GAT()
