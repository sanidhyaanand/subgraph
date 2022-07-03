import argparse
import os.path as osp
import time

import torch
from torch_geometric.loader import DataLoader

from tree_match.utils.dataset import SyntheticDataset, PairDataset
from tree_match.models.main_model import MainModel
from tree_match.models.gnn import GNN
from tree_match.utils.data import EarlyStoppingModule, save_initial_model

def train(av, model, batch_iter):
    device = "cuda:0" if av.has_cuda and av.want_cuda else "cpu"
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    # model = torch.nn.DataParallel(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), av.LR)
    es = EarlyStoppingModule()

    run = 0
    while run<av.NUM_RUNS: #or av.RUN_TILL_ES:
        print("Epoch {}".format(run))
        model.train()
        epoch_start_time = time.time()
        # Iterate through all batches
        epoch_loss = 0
        while True:
            batch_start_time = time.time()
            try:
                batch = next(batch_iter)
            except StopIteration:
                break
            batch = batch.to(device)
            if device == "cuda:0":
                batch.edge_index_s, batch.edge_index_t = batch.edge_index_s.type(torch.cuda.LongTensor), batch.edge_index_t.type(torch.cuda.LongTensor) 
            else:
                batch.edge_index_s, batch.edge_index_t = batch.edge_index_s.type(torch.LongTensor), batch.edge_index_t.type(torch.LongTensor)
            y = batch.y
            optimizer.zero_grad()
            S = model(batch.x_s, batch.edge_index_s, batch.x_s_batch, 
                            batch.x_t, batch.edge_index_t, batch.x_t_batch)
            loss = model.loss(S, y, batch.x_s_batch)
            loss = torch.mean(loss)
            acc = model.acc(S, y, batch.x_s_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += torch.sum(loss)
            print("Time taken to process batch: {}".format(time.time()-batch_start_time))
            print("Batch loss: {}".format(loss))
        run += 1
        print("Time taken to process epoch: {}".format(time.time()-epoch_start_time))
        print("Loss for current epoch: {}".format(epoch_loss))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('DIR_PATH', type=str)
    parser.add_argument('has_cuda', type=bool)
    parser.add_argument('want_cuda', type=bool)
    parser.add_argument('LR', type=float)
    parser.add_argument('NUM_RUNS', type=int)
    av = parser.parse_args()
    gnn = GNN(100, [32,16,16], 3)
    model = MainModel(gnn)
    save_initial_model(path=osp.join(av.DIR_PATH), name='ablation.pkl', model=model)

    for prob in [0.4, 0.1, 0.2, 0.3, 0.0, 0.5]:
        dataset = PairDataset(SyntheticDataset(root=osp.join(av.DIR_PATH, 'source')), 
                            SyntheticDataset(root=osp.join(av.DIR_PATH, 'target/'+str(prob))))
        loader = DataLoader(dataset, 32, follow_batch=['x_s', 'x_t'])
        batch_iter = iter(loader)
        train(av, model, batch_iter)

        


    