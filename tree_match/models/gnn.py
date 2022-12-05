import torch
from torch_geometric.nn.conv import GCNConv

class GNN(torch.nn.Module):
    def __init__(self, input_dim : int, filters : list[int]):
        super(GNN, self).__init__()
        self._in = input_dim
        self.filter_sizes = filters
        self.convs = torch.nn.ModuleList()
        num_layers = len(filters)

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
        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if idx != len(self.convs)-1:
                x = x.relu()
                x = torch.nn.functional.dropout(x, p=0.5, training=self.training)

        return x

    def __repr__(self):
        return ('{}({}, {}, num_layers={})').format(self.__class__.__name__, self._in,
                                                    self.filter_sizes, self.num_layers)
        