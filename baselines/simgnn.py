from typing import Optional, List

import torch
from torch.functional import Tensor
from torch_geometric.nn.conv import GCNConv, SAGEConv, GATConv

class GlobalContextAttention(torch.nn.Module):
    r"""
    Attention Mechanism layer for the attention operator from the 
    `"SimGNN: A Neural Network Approach to Fast Graph Similarity Computation"
    <https://arxiv.org/pdf/1808.05689.pdf>`_ paper
    TODO: Include latex formula for attention computation and aggregation update
    Args:
        input_dim: Input Dimension of the Node Embeddings
        activation: The Activation Function to be used for the Attention Layer
        activation_slope: Slope of the -ve part if the activation is Leaky ReLU
    """
    def __init__(self, input_dim, activation: str = "tanh", activation_slope: Optional[float] = None):
        super(GlobalContextAttention, self).__init__()
        self.input_dim = input_dim
        self.activation = activation 
        self.activation_slope = activation_slope
        
        self.initialize_parameters()
        self.reset_parameters()

    def initialize_parameters(self):
        r"""
        Weight initialization depends upon the activation function used.
        If ReLU/Leaky ReLU : He (Kaiming) Initialization
        If tanh/sigmoid : Xavier Initialization
        TODO: Initialisation methods need justification/reference
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.input_dim, self.input_dim))

    def reset_parameters(self):
        # BUG: ReLU needs an activation_slope, why? Presumably activation_slope was for leaky relu
        if self.activation == "leaky_relu" or self.activation == "relu":
            if self.activation_slope is None or self.activation_slope <= 0:
                raise ValueError(f"Activation function slope parameter needs to be a positive \
                                value. {self.activation_slope} is invalid")
            
            torch.nn.init.kaiming_normal_(self.weight_matrix, a = self.activation_slope, nonlinearity = self.activation)
        elif self.activation == "tanh" or self.activation == "sigmoid":
            torch.nn.init.xavier_normal_(self.weight_matrix)
        else:
            raise ValueError("Activation can only take values: 'relu', 'leaky_relu', 'sigmoid', 'tanh';\
                            {} is invalid".format(self.activation))

    def forward(self, x: Tensor):
        r""" 
        Args:
            x (torch.Tensor) : Node Embedding Tensor of shape N x D.
        
        Returns:
            representation (torch.Tensor): Global graph representation for input node 
            representation set.
        """
        if x.shape[1] != self.input_dim:
            raise RuntimeError("dim 1 of input tensor does not match dimension of weight matrix")
        # XXX: Have these dicts stored in separate files?
        activations = {"tanh": torch.nn.functional.tanh, "leaky_relu": torch.nn.functional.leaky_relu,
                        "relu": torch.nn.functional.relu, "sigmoid": torch.nn.functional.sigmoid}
        if self.activation not in activations.keys():
            raise ValueError(f"Invalid activation function specified: {self.activation}")

        # Generating the global context vector
        global_context = torch.mean(torch.matmul(x, self.weight_matrix), dim = 0)

        # Applying the Non-Linearity over global context vector
        _activation = activations[self.activation]
        global_context = _activation(global_context)

        # Computing attention weights and att-weight-aggregating node embeddings
        att_weights = torch.sigmoid(torch.matmul(x, global_context.view(-1, 1)))
        representation = torch.sum(x * att_weights, dim = 0)
        
        return representation

class NeuralTensorNetwork(torch.nn.Module):
    r"""
    Neural Tensor Network layer from the `"SimGNN: A Neural Network 
    Approach to Fast Graph Similarity Computation"
    <https://arxiv.org/pdf/1808.05689.pdf>`_ paper
    TODO: Include latex formula for NTN interaction score computation
    Args:
        input_dim: Input dimension of the graph-level embeddings
        slices: Number of slices (K) the weight tensor possesses. Often 
        interpreted as the number of entity-pair (in this use case - pairwise
        node) relations the data might possess.
        activation: Non-linearity applied on the computed output of the layer
    """
    def __init__(self, input_dim: int, slices: int = 16, activation: str = "tanh"):
        super(NeuralTensorNetwork, self).__init__()
        self.input_dim = input_dim
        self.slices = slices # K: hyperparameter
        self.activation = activation

        self.initialize_parameters()
        self.reset_parameters()

    def initialize_parameters(self):
        # XXX: Will arranging weight tensor as (k, d, d) cause problems in batching at dim 0?
        self.weight_tensor = torch.nn.Parameter(torch.Tensor(self.slices, self.input_dim, self.input_dim))
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.slices, 2 * self.input_dim))
        self.bias = torch.nn.Parameter(torch.Tensor(self.slices, 1))
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_tensor)
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.bias)
    
    def forward(self, h_i: Tensor, h_j: Tensor):
        r"""
        Args:
            h_i: First graph-level embedding
            h_j: Second graph-level embedding
        Returns:
            scores: (K, 1) graph-graph interaction score vector
        """
        scores = torch.matmul(h_i, self.weight_tensor)
        scores = torch.matmul(scores, torch.t(h_j)).squeeze(-1)
        scores += torch.matmul(self.weight_matrix, torch.t(torch.cat([h_i, h_j], dim=-1)))
        scores += self.bias
        
        # TODO: need to remove this from here and include it in a function in utils
        if self.activation == "tanh":
            _activation = torch.nn.functional.tanh
        elif self.activation == "sigmoid":
            _activation = torch.nn.functional.sigmoid
        
        scores = _activation(scores)

        return scores

class SimGNN(torch.nn.Module):
    r"""
    End to end implementation of SimGNN from the `"SimGNN: A Neural Network Approach
    to Fast Graph Similarity Computation" <https://arxiv.org/pdf/1808.05689.pdf>`_ paper.
    
    TODO: Provide description of implementation and differences from paper if any
    """
    def __init__(self, input_dim: int, ntn_slices: int = 16, filters: list = [64, 32, 16],
                 mlp_neurons: List[int] = [32,16,8,4], hist_bins: int = 16, conv: str = "gcn", 
                 activation: str = "tanh", activation_slope: Optional[float] = None, 
                 include_histogram: bool = False):
        # TODO: give a better name to the include_histogram flag 
        super(SimGNN, self).__init__()
        self.input_dim = input_dim
        self.ntn_slices = ntn_slices
        self.filters = filters
        self.mlp_neurons = mlp_neurons
        self.hist_bins = hist_bins
        self.conv_type = conv
        self.activation = activation
        self.activation_slope = activation_slope
        self.include_histogram = include_histogram

        self.setup_layers()
        self.reset_parameters()

    def setup_layers(self):
        # XXX: Should MLP and GNNs be defined as separate classes to avoid clutter?

        # Convolutional GNN layer
        self.convs = torch.nn.ModuleList()
        conv_methods = {"gcn": GCNConv, "sage": SAGEConv, "gat": GATConv}
        _conv = conv_methods[self.conv_type]
        num_layers = len(self.filters)
        self._in = self.input_dim
        for i in range(num_layers):
            self._out = self.filters[i]
            self.convs.append(_conv(in_channels=self._in, out_channels=self._out))
            self._in = self._out

        # Global self attention layer
        self.attention_layer = GlobalContextAttention(self.input_dim, activation = self.activation, 
                                                      activation_slope=self.activation_slope)
        # Neural Tensor Network module
        self.ntn_layer = NeuralTensorNetwork(self.input_dim, slices = self.ntn_slices, activation = self.activation)
        
        # MLP layer
        self.mlp = torch.nn.ModuleList()
        num_layers = len(self.mlp_neurons)
        if self.include_histogram:
            self._in = self.ntn_slices + self.hist_bins
        else: 
            self._in = self.ntn_slices
        for i in range(num_layers):
            self._out = self.mlp_neurons[i]
            self.mlp.append(torch.nn.Linear(self._in, self._out))
            self._in = self._out
        self.scoring_layer = torch.nn.Linear(self.mlp_neurons[-1], 1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.attention_layer.reset_parameters()
        self.ntn_layer.reset_parameters()
        for lin in self.mlp:
            lin.reset_parameters()
        
    def forward(self, x_i: Tensor, edge_index_i: Tensor, x_j: Tensor, edge_index_j: Tensor,
                conv_dropout: int = 0):
        r"""
        """
        # Strategy One: Graph-Level Embedding Interaction
        for filter_idx, conv in enumerate(self.convs):
            x_i = conv(x_i, edge_index_i)
            x_j = conv(x_j, edge_index_j)
            
            if filter_idx == len(self.convs) - 1:
                break
            x_i = torch.nn.functional.relu(x_i)
            x_i = torch.nn.functional.dropout(x_i, p = conv_dropout, training = self.training)
            x_j = torch.nn.functional.relu(x_j)
            x_j = torch.nn.functional.dropout(x_j, p = conv_dropout, training = self.training)

        h_i = self.attention_layer(x_i)
        h_j = self.attention_layer(x_j)

        interaction = self.ntn_layer(h_i, h_j) 
        
        # Strategy Two: Pairwise Node Comparison
        if self.include_histogram:
            sim_matrix = torch.matmul(h_i, h_j.transpose(-1,-2)).detach()
            sim_matrix = torch.sigmoid(sim_matrix)
            # XXX: is this if statement necessary? Can writing the histogram operation as a single 
            # tensor operation not accomodate batching?
            if len(sim_matrix.shape) == 3:
                scores = sim_matrix.view(sim_matrix.shape[0], -1, 1)
                hist = torch.cat([torch.histc(x, bins = self.hist_bins).unsqueeze(0) for x in scores], dim=0)
            else:
                scores = sim_matrix.view(-1, 1)
                hist = torch.histc(scores, bins = self.hist_bins)
            hist = hist.unsqueeze(-1)
            interaction = torch.cat((interaction, hist), dim = -2)
        
        # Final interaction score prediction
        for layer_idx, lin in enumerate(self.mlp):
            interaction = lin(interaction)
            interaction = torch.nn.functional.relu(interaction)
        # XXX: should torch.sigmoid be used for normalization of scores?
        interaction = self.scoring_layer(interaction)
        
        return interaction
    
    def loss(self, sim, gt):
        num_graph_pairs = sim.shape[-1] # Batch size

        batch_loss = torch.div(torch.sum(torch.square(sim-gt), dim=-1), num_graph_pairs)

        return batch_loss
    
