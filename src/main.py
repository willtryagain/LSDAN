import torch
from torch import nn
import torch.optim as optim
from GAT import GAT
from utils import *
from data_loading import parse_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "num_of_layers": 2,  # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
    "num_heads_per_layer": [8, 1],
    "num_features_per_layer": [CORA_NUM_INPUT_FEATURES, 8, CORA_NUM_CLASSES],
    "add_skip_connection": False,  # hurts perf on Cora
    "bias": True,  # result is not so sensitive to bias
    "dropout": 0.6,  # result is sensitive to dropout
    "layer_type": LayerType.IMP3,  # fastest implementation enabled by default
    "num_of_epochs": 10,
}

gat = GAT(
    num_of_layers=config['num_of_layers'],
    num_heads_per_layer=config['num_heads_per_layer'],
    num_features_per_layer=config['num_features_per_layer'],
    add_skip_connection=config['add_skip_connection'],
    bias=config['bias'],
    dropout=config['dropout'],
    layer_type=config['layer_type'],
    log_attention_weights=False  # no need to store attentions, used only in playground.py for visualizations
).to(device)

loss_fn = nn.CrossEntropyLoss(reduction='mean')
optimizer =  optim.Adam(gat.parameters(), lr=1e-4)

for epoch in range(config['num_of_epochs']):
    