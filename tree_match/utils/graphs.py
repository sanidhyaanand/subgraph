import torch
import numpy as np

class Graph(object):
    def __init__(self, edge_index):
        self.edge_index = edge_index
        self.adj_list = None
    
    @property
    def num_nodes(self):
        if self.adj_list is not None:
            return len(self.adj_list)
        else:
            return self.edge_index[0][-1].item() + 1
        
    @property
    def num_edges(self):
        if self.adj_list is not None:
            return np.sum([len(edge_list) for edge_list in self.adj_list])
        else:
            return self.edge_index.shape[1]

    def edgelist_to_adjlist(self) -> list():
        adj_list = []
        prev = 0
        curr_edge_list = []
        
        # Iterate over all edges
        for idx, (n_i, n_j) in enumerate(zip(self.edge_index[0],self.edge_index[1])):
            # If we are still looking at the same origin node as previous edge
            if prev == n_i:
                # Append new terminal node to current edge list
                curr_edge_list.append(int(n_j.item()))
                # If this is the last edge
                if idx == len(self.edge_index[0]) - 1:
                    # Append current edge list to adjacency list
                    adj_list.append(curr_edge_list)
            # If we encounter a new origin node
            else:
                # Append the current edge list to adjacency list
                adj_list.append(curr_edge_list)
                # Initialise current edge list for new origin node
                curr_edge_list = []
                # Append new terminal node to current edge list
                curr_edge_list.append(int(n_j.item()))
                # If this is the last edge
                if idx == len(self.edge_index[0]) - 1:
                    adj_list.append(curr_edge_list)
                prev = n_i
        return adj_list

    def process(self):
        self.adj_list = self.edgelist_to_adjlist()
        
    def adjlist_to_edgelist(self) -> torch.tensor:
        edges = torch.tensor([[],[]])
        for idx, edge_list in enumerate(self.adj_list):
            to_attach = torch.cat((torch.tensor([[idx] * len(edge_list)]), 
                                   torch.tensor([edge_list])), axis = 0)
            edges = torch.cat((edges, to_attach), axis = 1)
        return edges

class EdgeDropper(Graph):
    def __init__(self, edge_index, drop_prob):
        super().__init__(edge_index)
        self.process()
        self.drop_prob = drop_prob
    
    def DFS(self, node_idx):
        self.visited[node_idx] = True
        edges = self.adj_list[node_idx]

        i = 0
        while i < len(edges):
            if (not self.visited[edges[i]]):
                self.DFS(edges[i])
            i += 1

    def isConnected(self, num_nodes):
        # Returns true if given graph is
        # connected, else false
        self.visited = [False] * num_nodes
        
        # Find all reachable vertices
        # from first vertex
        self.DFS(0)
        
        # If set of reachable vertices
        # includes all, return true.
        for i in range(1, num_nodes):
            if (self.visited[i] == False):
                return False
        
        return True

    # This function assumes that edge 
    # (u, v) exists in graph or not,
    def isBridge(self, u, v, checked_nodes):
        if (u,v) in checked_nodes or (v,u) in checked_nodes:
            return True
        # Remove edge from undirected graph
        idx = np.where(np.array(self.adj_list[u]) == v)[0]
        if len(idx) == 1:
            idx = idx[0]
        else: 
            raise ValueError('Edges are repeated in adjacency list derived from edge_index')
        self.adj_list[u].pop(idx)

        res = self.isConnected(len(self.adj_list))
        self.adj_list[u].insert(idx, v)
        checked_nodes.append((u,v))

        # Return true if graph becomes
        # disconnected after removing
        # the edge.
        return (res == False)

    def get_target_graph(self):
        # node_ids = range(torch.max(edge_index[0]))
        checked_nodes = []
        for n_i, edge_nodes in enumerate(self.adj_list):
            # List of boolean values for each edge
            bridges = [self.isBridge(n_i, n_j, checked_nodes) for n_j in edge_nodes]
            drop_val = [(np.random.binomial(1,self.drop_prob) and not val) for val in bridges]
            new_edge_nodes = [edge_nodes[idx] for idx in np.where(np.array(drop_val) == 0)[0]]
            self.adj_list[n_i] = new_edge_nodes

        return self.adjlist_to_edgelist()