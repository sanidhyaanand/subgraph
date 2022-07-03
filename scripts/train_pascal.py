import argparse
import os.path as osp
import time

import torch
from torch_geometric.loader import DataLoader

from tree_match.utils.dataset import PairDataset, load_dataset
from tree_match.models.main_model import MainModel
from tree_match.models.gnn import GNN
from tree_match.utils.data import EarlyStoppingModule, PairData, save_initial_model, shuffle_graph

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
    _max = 0
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
            
        print(_max)
        run += 1
        print("Time taken to process epoch: {}".format(time.time()-epoch_start_time))
        print("Loss for current epoch: {}".format(epoch_loss))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('DIR_PATH', type=str) # /media/data/anands02dm_data/pascal_voc
    parser.add_argument('has_cuda', type=bool)
    parser.add_argument('want_cuda', type=bool)
    parser.add_argument('LR', type=float)
    parser.add_argument('NUM_RUNS', type=int)
    parser.add_argument('category', type=str)
    parser.add_argument('--seed', type=str, default=0)
    av = parser.parse_args()
    torch.manual_seed(seed=av.seed)
    gnn = GNN(100, [32,16,16], 3)
    model = MainModel(gnn)
    save_initial_model(path=av.DIR_PATH, name='pascal_voc', model=model)

    dataset_s = load_dataset(dpath=osp.join(av.DIR_PATH, av.category), name='pascal_voc', category=av.category, train=True)
    dataset = PairDataset(dataset_s, dataset_s, sample=True)
    loader = DataLoader(dataset, 32, follow_batch=['x_s', 'x_t'])
    batch_iter = iter(loader)
    train(av, model, batch_iter)