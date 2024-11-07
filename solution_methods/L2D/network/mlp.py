import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
        A MLP class with configurable layers and dimensions.

        Args:
        - num_layers (int): Number of hidden layers (excluding the input layer).
        - input_dim (int): The input feature dimension.
        - hidden_dim (int): The number of hidden units for each hidden layer.
        - output_dim (int): The output dimension.
        """

        super(MLP, self).__init__()

        if num_layers < 1:
            raise ValueError("The number of layers should be at least 1.")

        self.num_layers = num_layers

        if num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
            self.linear_or_not = True
        else:
            # Multi-layer model
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for i in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            self.linear_or_not = False

            for i in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            for i in range(self.num_layers - 1):
                x = F.relu(self.batch_norms[i](self.linears[i](x)))
            return self.linears[self.num_layers - 1](x)


class MLPActor(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
        Args:
        - num_layers (int): Number of hidden layers (excluding the input layer).
        - input_dim (int): The input feature dimension.
        - hidden_dim (int): The number of hidden units for each hidden layer.
        - output_dim (int): The output dimension, typically the action space size.
        """
        super(MLPActor, self).__init__()

        if num_layers < 1:
            raise ValueError("The number of layers should be at least 1.")

        self.num_layers = num_layers

        if num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
            self.linear_or_not = True
        else:
            self.linears = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for i in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            self.linear_or_not = False

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            for i in range(self.num_layers - 1):
                x = torch.tanh(self.linears[i](x))
            return self.linears[self.num_layers - 1](x)


class MLPCritic(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
        - num_layers (int): Number of hidden layers (excluding the input layer).
        - input_dim (int): The input feature dimension.
        - hidden_dim (int): The number of hidden units for each hidden layer.
        - output_dim (int): The output dimension, typically the value function size.
        """
        super(MLPCritic, self).__init__()

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")

        self.num_layers = num_layers

        if self.num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
            self.linear_or_not = True
        else:
            self.linears = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for i in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            self.linear_or_not = False

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            for i in range(self.num_layers - 1):
                x = torch.tanh(self.linears[i](x))
            return self.linears[self.num_layers - 1](x)