import torch
from torch_geometric.data import Data

from tree_match.models.main_model import MainModel
from tree_match.models.gnn import GNN

def main():
    x_s = torch.tensor([[0,0,0,1],
                        [0,0,1,0],
                        [0,0,1,0],
                        [0,1,0,0]], dtype=torch.float)
    e_s = torch.tensor([[0,0,0,1,1,2,2,3],
                        [1,2,3,0,2,0,1,0]])
    batch_s = torch.tensor([0,0,0,0])
    x_t = torch.tensor([[0,0,0,1],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,1,0]], dtype=torch.float)
    e_t = torch.tensor([[0,0,0,1,2,2,3,3],
                        [1,2,3,0,0,3,0,2]])
    batch_t = torch.tensor([0,0,0,0])
    
    psi_1 = GNN(input_dim=4, filters=[32,16,8])
    model = MainModel(psi_1=psi_1)
    S = model(x_s, e_s, batch_s, x_t, e_t, batch_t, include_gnn = False, num_iter=5)

main()