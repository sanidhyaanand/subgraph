from tree_match.models.gnn import GNN
import torch
from torch_geometric.data import Data

features = torch.tensor([[3],[1],[1],[1]], dtype = torch.float)
edge_index = torch.tensor([[0,0,0,1,2,3],[1,2,3,0,0,0]], dtype = torch.long)

x = Data(x=features, edge_index=edge_index)

conv_layers = GNN(input_dim=x.x.shape[1], filters=[64,32,16], num_layers=3)

fpass = conv_layers(x.x, x.edge_index)
print(fpass, conv_layers)