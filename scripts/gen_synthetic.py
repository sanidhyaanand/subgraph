'''
Module to generate synthetic Erdos Renyi graphs, treated
as query graphs in datasets. Graphs generated are undirected 
until specified otherwise

TODO:
1. Need to seed the torch random generator for control
2. 
'''
import os.path as osp
import os
import sys
sys.path.append('/export/home/anands02dm/subgraph')
import argparse

import numpy as np
import torch
from torch_geometric.utils.random import erdos_renyi_graph

from tree_match.utils.dataset import SyntheticDataset
from tree_match.utils.data import save_data

def generate_graph(num_nodes, edge_prob):
    graph = erdos_renyi_graph(num_nodes, edge_prob)
    if graph.nelement() == 0:
        generate_graph(num_nodes, edge_prob)
    return graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--EDGE_PROB', type = float, default = 0.20)
    parser.add_argument('--GRAPH_SIZE', type = int, default = 50)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--DIR_PATH', type = str)
    av = parser.parse_args()

    paths = [osp.join(av.DIR_PATH), 
        osp.join(av.DIR_PATH, 'raw'),
        osp.join(av.DIR_PATH, 'processed')]
    for path in paths:
        if not osp.exists(path):
            os.makedirs(path)

    num_graphs = 1000
    torch.manual_seed(av.seed)
    np.random.seed(av.seed)

    graph_size = av.GRAPH_SIZE
    edge_prob = av.EDGE_PROB

    max_idx = 0
    for graph_idx in range(num_graphs):
        gen_graph = generate_graph(graph_size, edge_prob)
        node_idxs = torch.unique(gen_graph[0]).type(torch.int)
        if len(node_idxs) != node_idxs[-1].item() + 1:
            change = {}
            missing = []
            for idx in range(0, len(node_idxs) - 1):
                if node_idxs[idx].item() != node_idxs[idx+1].item() - 1:
                    missing.append(node_idxs[idx].item() + 1)

            for elem in missing:
                count = -1
                for i in range(elem+1, len(node_idxs)+len(missing)):
                    change[i] = i + count
                count -= 1
            for idx, (elem1, elem2) in enumerate(zip([int(elem.item()) for elem in list(gen_graph[0])], 
                                                    ([int(elem.item()) for elem in list(gen_graph[1])]))):
                if elem1 in change.keys():
                    gen_graph[0][idx] = change[elem1]
                if elem2 in change.keys():
                    gen_graph[1][idx] = change[elem2]

        max_node_idx = torch.max(gen_graph[0].type(torch.int))
        node_degrees = torch.bincount(gen_graph[0].type(torch.int))
        # node_matrix = torch.zeros([max_node_idx+1, torch.max(node_degrees)])
        node_matrix = torch.zeros([max_node_idx+1, av.GRAPH_SIZE])
        for idx, degree in enumerate(node_degrees):
                node_matrix[idx][degree-1] = 1
        save_data(osp.join(av.DIR_PATH, 'raw'), (node_matrix, gen_graph), 'graph', graph_idx)
        max_idx = max_node_idx

    dataset = SyntheticDataset(root = av.DIR_PATH)

        