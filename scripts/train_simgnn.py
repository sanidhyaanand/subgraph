import argparse
import os.path as osp
import time

import torch
from torch_geometric.loader import DataLoader

from tree_match.utils.dataset import SyntheticDataset, PairDataset
from baselines.simgnn import SimGNN
from tree_match.utils.data import EarlyStoppingModule, save_initial_model

def train(av, model, loader, log_string, batch_size=32, num_iter=1):
    device = "cuda:0" if av.has_cuda and av.want_cuda else "cpu"
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    # model = torch.nn.DataParallel(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), av.LR, weight_decay=av.weight_decay)
    es = EarlyStoppingModule()
    f = open(log_string, mode='w')

    print(f"Ablation graph data with num_nodes={av.num_nodes}, edge_prob={av.edge_prob}, drop_prob={prob}\n", file=f)

    print("Hyperparameters -----------------------------------", file=f)
    print(f"Learning rate: {av.LR}", file=f)
    print(f"L2 regulariser Lambda: {av.weight_decay}\n", file=f)
    print(f"Number of sinkhorn module iterations: {num_iter}\n", file=f)
    run = 0
    while run<av.NUM_RUNS: #or av.RUN_TILL_ES:
        batch_iter = iter(loader)
        print("Epoch {} -------------------------------------\n".format(run), file=f)
        model.train()
        epoch_start_time = time.time()
        # Iterate through all batches
        epoch_loss = []
        for batch_idx in range(batch_size):
            batch_start_time = time.time()
            try:
                batch = next(batch_iter)
            except StopIteration:
                break
            print('Processing batch '+f'{batch_idx}', file=f)
            batch = batch.to(device)
            if device == "cuda:0":
                batch.edge_index_s, batch.edge_index_t = batch.edge_index_s.type(torch.cuda.LongTensor), batch.edge_index_t.type(torch.cuda.LongTensor) 
            else:
                batch.edge_index_s, batch.edge_index_t = batch.edge_index_s.type(torch.LongTensor), batch.edge_index_t.type(torch.LongTensor)
            y = batch.y
            
            # Setting batch gradients to zero
            optimizer.zero_grad()

            # Forward Pass
            interaction = model(batch.x_s, batch.edge_index_s, batch.x_t, batch.edge_index_t)
            
            # Calculating batch loss and accuracy
            loss = model.loss(interaction, y, batch.x_s_batch, reduction="sum")
            # acc = model.acc(interaction, y, batch.x_s_batch)
            acc = None
            
            # Calculating gradient and weight change
            loss.backward()
            optimizer.step()
            
            epoch_loss.append(loss)
            batch_idx+=1

            print(f'Finished batch {batch_idx}')
            print("Time taken to process batch: {}".format(time.time()-batch_start_time), file=f)
            print("Batch loss: {}".format(loss), file=f)
            print(f'Average Batch accuracy: {acc}\n', file=f)
        run += 1
        print(f'Finished epoch {run}')
        print("Time taken to process epoch: {}".format(time.time()-epoch_start_time), file=f)
        print("Loss for current epoch: {}\n".format(sum(epoch_loss)), file=f)
    f.close()
    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('DIR_PATH', type=str)
    parser.add_argument('has_cuda', type=bool)
    parser.add_argument('want_cuda', type=bool)
    parser.add_argument('LR', type=float)
    parser.add_argument('weight_decay', type = float)
    parser.add_argument('NUM_RUNS', type=int)
    parser.add_argument('num_nodes', type=int)
    parser.add_argument('edge_prob', type=float)
    av = parser.parse_args()

    # python -m scripts.train_simgnn /media/data/anands02dm_data/synthetic_er/50/ True True 1.0 0.1 1 50 0.2

    torch.manual_seed(0)
    # save_initial_model(path=av.DIR_PATH, name='ablation.pkl', model=model)

    for prob in [0.1]:
        dataset = PairDataset(SyntheticDataset(root=osp.join(av.DIR_PATH, 'source')), 
                            SyntheticDataset(root=osp.join(av.DIR_PATH, 'target/'+str(prob))))
        loader = DataLoader(dataset, 32, follow_batch=['x_s', 'x_t'])
        edge_prob_desc = str(int(av.edge_prob*10))
        prob_desc = str(int(prob*10))
        log_string = f"/media/data/anands02dm_data/synthetic_er/results/baselines/simgnn/log/{av.num_nodes}_{edge_prob_desc}_{prob_desc}.txt"
        # with open(log_string, 'r+') as f:
        #     f.truncate(0)

        model = SimGNN(input_dim=dataset[0].x_s.shape[-1])
        
        print("Logging to:", log_string)
        for num_iter in [1]:
            train(av, model, loader, log_string, 32, num_iter=num_iter)