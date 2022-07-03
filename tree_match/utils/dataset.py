import os.path as osp
import os
import random
import math

import re
import numpy as np
from torch_geometric.data import Dataset, Data
from torch_geometric.transforms import BaseTransform, Compose
import torch
import pickle

from .data import PairData

class Face2Edge(BaseTransform):
    def __call__(self, data):
        list_of_edges = []
        tri = data.face.t()
        
        def undirected_edge(a, b):
            return [[a,b], [b,a]]

        for triangle in tri:
            for e1, e2 in [[0,1],[1,2],[2,0]]: # for all edges of triangle
                list_of_edges.extend(undirected_edge(triangle[e1],triangle[e2])) # always lesser index first

        edge_index = np.unique(list_of_edges, axis=0).T # remove duplicates
        data.edge_index = torch.from_numpy(edge_index).to(device=data.face.device, dtype=torch.long)
        del data.face

        return data

# def process_pascalvoc(dataset, seed):
#     torch.manual_seed(seed)
#     random.seed(seed)
#     from torch_geometric.data import DataListLoader

#     data_list = DataListLoader(dataset, 32, follow_batch=['x_s', 'x_t'])
#     for batch in data_list:
#         batch.x_s = batch.x
#         batch.e_s = batch.edge_index
#         target_perm = 

def load_dataset(dpath: str, name: str, category = None, train: bool = True):
    if not osp.isdir(dpath):
        os.makedirs(dpath)
    fname = osp.join(dpath, name + ".pkl")
    if os.path.isfile(fname):
        d = pickle.load(open(fname,"rb"))
    else:
        if name == "pascal_voc":
            from torch_geometric.datasets import PascalVOCKeypoints
            from torch_geometric.transforms import Delaunay

            transform = Compose([Delaunay(), Face2Edge()])
            assert category is not None, "Need to specify category for PascalVOCKeypoints"
            if train:
                d = PascalVOCKeypoints(root=dpath, category=category, train=True, transform=transform)
            else:
                d = PascalVOCKeypoints(root=dpath, category=category, train=False, transform=transform)
        with open(fname,"wb") as f:
            pickle.dump(d,f)
    return d

class SyntheticDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        with os.scandir(self.raw_dir) as files:
            return [file for file in files]

    @property
    def processed_file_names(self):
        with os.scandir(self.processed_dir) as files:
            return [file for file in files]
    
    def len(self):
        return len(self.processed_file_names) - 2

    def process(self):
        '''
        Reads graph data tuples saved in 'root/raw' directory 
        and processes and saves them as Data objects in
        'root/processed' directory.
        '''
        idx = 0
        paths = [x.split('.')[-2] for x in self.raw_paths]
        path_idxs = np.argsort(np.array([int(x.split('_')[-1]) for x in paths]))
        paths = [self.raw_paths[idx] for idx in path_idxs]

        for raw_path in paths:
            # Read data from `raw_path`.
            raw_data = torch.load(raw_path)
            if len(raw_data) == 3:
                data = Data(x = raw_data[0], edge_index = raw_data[1], y = raw_data[2])
            else:
                data = Data(x = raw_data[0], edge_index = raw_data[1])

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'graph_{idx}.pt'))
            idx += 1

    def get(self, idx):
        '''
        Returns graph with index `idx` from self.root_dir
        '''
        graph = torch.load(osp.join(self.processed_dir, f'graph_{idx}.pt'))
        return graph

class PairDataset(torch.utils.data.Dataset):
    r"""Combines two datasets, a source dataset and a target dataset, by
    building pairs between separate dataset examples.
    Args:
        dataset_s (torch.utils.data.Dataset): The source dataset.
        dataset_t (torch.utils.data.Dataset): The target dataset.
        sample (bool, optional): If set to :obj:`True`, will sample exactly
            one target example for every source example instead of holding the
            product of all source and target examples. (default: :obj:`False`)
    """
    def __init__(self, dataset_s, dataset_t, sample=False):
        self.dataset_s = dataset_s
        self.dataset_t = dataset_t
        self.sample = sample

    def __len__(self):
        return len(self.dataset_s) if self.sample \
            else len(self.dataset_s) * len(self.dataset_t)

    def __getitem__(self, idx):
        if self.sample:
            data_s = self.dataset_s[idx]
            data_t = self.dataset_t[np.random.randint(0, len(self.dataset_t) - 1)]
        else:
            data_s = self.dataset_s[idx]
            data_t = self.dataset_t[idx]
        
        if data_s.y is not None:
            # if len(data_s.y) != len(data_t.y):
            #     diff = abs(len(data_s.y) - len(data_t.y))
            #     pad = -1 * torch.ones(diff, dtype=torch.long, device=data_s.y.device)
            #     if len(data_s.y) > len(data_t.y):
            #         data_t.y = torch.cat([data_t.y, pad], dim=0)
            #     else:
            #         data_s.y = torch.cat([data_s.y, pad], dim=0)
            ys = []
            for item in data_s.y:
                if item not in data_t.y:
                    ys.append(-1)
                else:
                    ys.append(torch.where(data_t.y == item)[0].item())
            ys = torch.tensor(ys, device=data_s.y.device)

        return PairData(
            x_s=data_s.x,
            edge_index_s=data_s.edge_index,
            x_t=data_t.x,
            edge_index_t=data_t.edge_index,
            y=torch.stack((data_s.y, ys), dim=1)
        )

    def __repr__(self):
        return '{}({}, {}, sample={})'.format(self.__class__.__name__,
                                              self.dataset_s, self.dataset_t,
                                              self.sample)