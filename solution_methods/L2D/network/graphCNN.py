import torch
import torch.nn as nn
import torch.nn.functional as F
from solution_methods.L2D.network.mlp import MLP


class GraphCNN(nn.Module):
    def __init__(self,
                 num_layers,
                 num_mlp_layers,
                 input_dim,
                 hidden_dim,
                 # final_dropout,
                 learn_eps,
                 neighbor_pooling_type,
                 device):
        """
        A Graph Convolutional Network (GCN) implementation with custom pooling and MLP layers.

        Args:
        - num_layers (int): Total number of layers in the network (including input layer).
        - num_mlp_layers (int): Number of layers in each MLP (excluding the input layer).
        - input_dim (int): Dimensionality of input features.
        - hidden_dim (int): Dimensionality of hidden layers.
        - learn_eps (bool): If True, learn epsilon to distinguish center nodes from neighboring nodes.
        - neighbor_pooling_type (str): Type of pooling for neighbors (options: 'max', 'sum', 'average').
        - device (torch.device): The device (CPU/GPU) for model operations.
        """
        super(GraphCNN, self).__init__()

        # Initialize device and network properties
        self.device = device
        self.num_layers = num_layers
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps

        # List of MLPs and batch normalization layers
        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        # generate MLP layers (apart from output)
        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def next_layer_eps(self, h, layer, padded_neighbor_list = None, adj_block = None):
        # pooling neighboring nodes and center nodes separately by epsilon reweighting.

        if self.neighbor_pooling_type == "max":
            pooled = self.maxpool(h, padded_neighbor_list)
        else: # sum or average
            pooled = torch.mm(adj_block, h)
            if self.neighbor_pooling_type == "average":
                degree = torch.mm(adj_block, torch.ones((adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree

        # Re-weight the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer])*h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        return F.relu(h)

    def next_layer(self, h, layer, padded_neighbor_list = None, adj_block = None):
        # Pooling neighboring nodes and center nodes altogether

        if self.neighbor_pooling_type == "max":
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            pooled = torch.mm(adj_block, h)
            if self.neighbor_pooling_type == "average":
                degree = torch.mm(adj_block, torch.ones((adj_block.shape[0], 1)).to(self.device))
                pooled = pooled / degree

        # representation of neighboring and center nodes
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        return F.relu(h)

    def forward(self, x, graph_pool, padded_nei, adj):

        h = x
        padded_neighbor_list = padded_nei if self.neighbor_pooling_type == "max" else None
        adj_block = adj if self.neighbor_pooling_type != "max" else None

        for layer in range(self.num_layers-1):
            if self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, adj_block=adj_block)
            elif self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, adj_block=adj_block)

        h_nodes = h.clone()
        pooled_h = torch.sparse.mm(graph_pool, h)

        return pooled_h, h_nodes