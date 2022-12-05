import argparse
import sys
sys.path.append('/export/home/anands02dm/subgraph')
import os.path as osp
import os

import numpy as np
import torch

from tree_match.utils.dataset import SyntheticDataset
from tree_match.utils.graphs import EdgeDropper
from tree_match.utils.data import save_data, shuffle_graph

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type = int, default = 42)
parser.add_argument('--num_graphs', type = int, default = 1000)
parser.add_argument('--drop_prob', type = float)
parser.add_argument('--SOURCE_DIR_PATH', type = str)
parser.add_argument('--TARGET_DIR_PATH', type = str)
av = parser.parse_args()

source_dataset = SyntheticDataset(av.SOURCE_DIR_PATH)

np.random.seed(av.seed)
torch.manual_seed(av.seed)

paths = [osp.join(av.TARGET_DIR_PATH, str(av.drop_prob)), 
        osp.join(av.TARGET_DIR_PATH, str(av.drop_prob), 'raw'),
        osp.join(av.TARGET_DIR_PATH, str(av.drop_prob), 'processed')]
for path in paths:
    if not osp.exists(path):
        os.makedirs(path)

for idx in range(av.num_graphs):
    source_graph = source_dataset.get(idx)
    dropper = EdgeDropper(source_graph.edge_index, av.drop_prob)
    s_idxs = torch.arange(source_graph.x.shape[0])
    perm = torch.randperm(source_graph.x.shape[0])
    
    target_x, target_e = shuffle_graph(
        source_graph.x,
        dropper.get_target_graph().type(torch.long), 
        perm
        )
    
    mapping = torch.sort(torch.stack((s_idxs, perm), dim=0)).indices[1]
    target_graph = (target_x, target_e.to(dtype=torch.long), torch.stack((s_idxs, mapping), dim=0))
    
    save_data(paths[1], target_graph, 'graph', idx)

target_dataset = SyntheticDataset(root = paths[0])
        

    