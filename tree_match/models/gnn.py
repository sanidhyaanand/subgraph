import torch
from torch_geometric.nn.conv import GCNConv

class GNN(torch.nn.Module):
    def __init__(self, input_dim : int, filters : list[int], num_layers : int):
        super(GNN, self).__init__()
        self._in = input_dim
        self.filter_sizes = filters
        self.convs = torch.nn.ModuleList()
        self.num_layers = num_layers

        assert len(filters) == num_layers, "Length of filter dimension list does not\
                                            match number of layers"
        for i in range(num_layers):
            self._out = filters[i]
            self.convs.append(GCNConv(in_channels=self._in, out_channels=self._out))
            self._in = self._out

        self._in = input_dim
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)

        return x

    def __repr__(self):
        return ('{}({}, {}, num_layers={})').format(self.__class__.__name__, self._in,
                                                    self.filter_sizes, self.num_layers)
        