from torch_geometric.utils import erdos_renyi_graph, to_dense_batch
from tree_match.models.main_model import MainModel
from tree_match.models.gnn import GNN
from tree_match.utils.data import PairData
import torch
from torch_geometric.loader import DataLoader

edge_prob = 0.4
feature_dim = 6

def set_seed():
    torch.manual_seed(42)

set_seed()

num_nodes = 10
edge_index_s = erdos_renyi_graph(num_nodes, edge_prob)
edge_index_t = erdos_renyi_graph(num_nodes, edge_prob)
x_s = torch.randn((num_nodes, feature_dim))
x_t = torch.randn((num_nodes, feature_dim))
y = torch.arange(num_nodes)
y = torch.stack([y, y], dim=1)
data1 = PairData(edge_index_s, x_s, edge_index_t, x_t, y)


num_nodes = 15
edge_index_s = erdos_renyi_graph(num_nodes, edge_prob)
edge_index_t = erdos_renyi_graph(num_nodes, edge_prob)
x_s = torch.randn((num_nodes, feature_dim))
x_t = torch.randn((num_nodes, feature_dim))
y = torch.arange(num_nodes)
y = torch.stack([y, y], dim=1)
data2 = PairData(edge_index_s, x_s, edge_index_t, x_t, y)

data_list = [data1,data2,data1,data2] # batch of graphs
def_batch_size = 3
psi_1 = GNN(input_dim=feature_dim, filters=[32,16,8], num_layers=3)

def test_batching(batch_size):
    print(
        '\nTESTING BATCHING PROCEDURE\n--------------------------------------------'
    )
    loader = DataLoader(data_list, batch_size = batch_size, follow_batch = ['x_s', 'x_t'])
    batch_iter = iter(loader)
    for _ in range(batch_size):
        try:
            batch = next(batch_iter)
        except StopIteration:
            break
        print("Currently processing batch:", batch)
        print("Number of nodes in batch graph:", batch.x_s_batch.shape)

def test_on_single_graphs(data):
    print(
        '\nTESTING MODEL ON SINGLE GRAPH OBJECT\n--------------------------------------------'
    )
    set_seed()
    assert len(data.x_s.shape) == 2, "Multiple graphs passed to test function"
    batch_s = torch.zeros(size=torch.Size([data.x_s.shape[0]]), dtype=torch.long)
    batch_t = torch.zeros(size=torch.Size([data.x_t.shape[0]]), dtype=torch.long)
    model = MainModel(psi_1=psi_1)

    S = model(data.x_s, data.edge_index_s, batch_s, data.x_t, data.edge_index_t, batch_t)
    acc = model.acc(S,data.y)
    print("Accuracy of model is:", acc)
    loss = model.loss(S,data.y)
    print("Loss in this run:", loss)

def test_on_multiple_graphs(data_list):
    print(
        '\nTESTING MODEL ON BATCHED DATA OBJECTS\n--------------------------------------------'
        )
    set_seed()
    loader = DataLoader(data_list, batch_size = def_batch_size, follow_batch = ['x_s', 'x_t'])
    model = MainModel(psi_1=psi_1)
    batch_iter = iter(loader)
    for _ in range(def_batch_size):
        try:
            batch = next(batch_iter)
        except StopIteration:
            break
        print("\nCurrently processing batch:", batch)
        assert batch.y.shape == torch.Size([batch.x_s.shape[0],2])
        assert torch.equal(batch.x_s_batch, batch.x_t_batch)

        # Batched along dim 0 
        S = model(batch.x_s, batch.edge_index_s, batch.x_s_batch, batch.x_t, batch.edge_index_t, batch.x_t_batch)
        assert S.shape == torch.Size([batch.x_s.shape[0], torch.max(torch.bincount(batch.x_t_batch))])
        print("Forward pass of model successful")
        acc = model.acc(S, batch.y, batch.x_s_batch)    
        print("Accuracy of model per batch is:", acc)
        loss = model.loss(S, batch.y, batch.x_s_batch)
        print("Loss in this run per batch:", loss)

test_batching(def_batch_size)
test_on_single_graphs(data1)
test_on_multiple_graphs(data_list)