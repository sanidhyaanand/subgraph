from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_scatter import scatter
from torch_geometric.nn.inits import reset

def masked_sinkhorn(src, mask, num_steps, tau, device, dim=-1):
    out = src.masked_fill(~mask, float('-inf'))
    out = torch.softmax(out, dim=dim)
    out = out.masked_fill(~mask, 0)
    out = torch.div(out, tau)
    
    cn_size, rn_size = list(out.shape), list(out.shape)
    cn_size[-1] = cn_size[-2]
    rn_size[-2] = rn_size[-1]
    c_norm = torch.ones(cn_size, device=device)
    r_norm = torch.ones(rn_size, device=device)

    for _ in range(num_steps):
        # col norm
        norm = c_norm @ out
        out = torch.div(out, norm + 1e-8)
        # r norm
        norm = out @ r_norm
        out = torch.div(out, norm + 1e-8)

    # Include not converged warning
    
    return out

def masked_softmax(src, mask, dim=-1):
    out = src.masked_fill(~mask, float('-inf'))
    out = torch.softmax(out, dim=dim)
    out = out.masked_fill(~mask, 0)

    return out

def to_sparse(x, mask):
    return x[mask]

class MainModel(torch.nn.Module):
    def __init__(self, psi_1, k=-1, lin_dim=16):
        super(MainModel, self).__init__()
        self.psi_1 = psi_1
        self.mlp = Sequential(
            Linear(1, lin_dim),
            ReLU(),
            Linear(lin_dim, 1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.psi_1.reset_parameters()
        reset(self.mlp)
    
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x_s, edge_index_s, batch_s,
                x_t, edge_index_t, batch_t, include_gnn = True, include_mlp = False,
                num_steps = 5, tau = 0.1, bypass = False, num_iter=1):
        r'''
        Args:
            x_s (Tensor): Batched features of source graph nodes of
                size :obj:`[batch_size * num_nodes, C_in]`, where
                :obj:`C_in` is the number of input node features.
            edge_index_s (LongTensor): Edge connectivities of source
                graph in COO format with shape :obj:`[2, num_edges]`.
            batch_s (LongTensor): Batch vector of source graph nodes
                indicating node to graph assignment, with shape
                :obj:`[batch_size * num_nodes]. Set to :obj:`None`
                for operating on single graphs.
            x_t (Tensor): Batched features of target graph nodes of
                size :obj:`[batch_size * num_nodes, C_in]`, where
                :obj:`C_in` is the number of input node features.
            edge_index_s (LongTensor): Edge connectivities of target
                graph in COO format with shape :obj:`[2, num_edges]`.
            batch_s (LongTensor): Batch vector of source graph nodes
                indicating node to graph assignment, with shape
                :obj:`[batch_size * num_nodes]. Set to :obj:`None`
                for operating on single graphs.
        '''
        if include_gnn:
            h_s = self.psi_1(x_s, edge_index_s)
            h_t = self.psi_1(x_t, edge_index_t)
        else:
            h_s = x_s
            h_t = x_t
        
        
        # Convert [sum(N_{i}) where i in range(B), C_out] tensor to [B = batch_size, N = max(N_{i}) where i in range(B), C_out] tensor
        h_s, s_mask = to_dense_batch(h_s, batch_s, fill_value=0) # [B, N_s = max_{s}(N_s(i)), C_out], [B, N_s]
        h_t, t_mask = to_dense_batch(h_t, batch_t, fill_value=0) # [B, N_t = max_{t}(N_t(i)), C_out], [B, N_t]
        adj_s = to_dense_adj(edge_index=edge_index_s, batch=batch_s)
        adj_t = to_dense_adj(edge_index=edge_index_t, batch=batch_t)

        sim_matrix = h_s @ h_t.transpose(-1,-2) # [B, N_s, N_t]    
        (B, N_s, N_t) = sim_matrix.size()
        S_mask = s_mask.view(B, N_s, 1) & t_mask.view(B, 1, N_t)

        sim_t_mask = t_mask.repeat_interleave(torch.bincount(batch_s), dim=0) # [sum{N_s(i)}, N_t], mask for the final similarity matrix
        
        # Detach neighbour sinkhorn iteration module
        if bypass: 
            return (sim_matrix[s_mask]).masked_fill(~sim_t_mask, 0)

        sim_update = torch.zeros(sim_matrix.shape, device=self.device)

        # Somehow tensorize this through masking?
        for _ in range(num_iter):
            for i in range(N_s):
                # make batch list of neighbours of i
                n_i = [x.nonzero().t()[0] for x in adj_s[:,i,:]]
                num_i = [len(x) for x in n_i]
                max_i = max(num_i)
                for j in range(N_t):
                    # batch list of neighbours of j
                    n_j = [x.nonzero().t()[0] for x in adj_t[:,j,:]]
                    num_j = [len(x) for x in n_j]
                    max_j = max(num_j)

                    S_hat = torch.zeros(size=(B, max_i, max_j), device=self.device)
                    S_hat_mask = torch.zeros(size=(B, max_i, max_j), device=self.device)
                    # Tensorize
                    for batch_idx, nei in enumerate(n_i):
                        nej = n_j[batch_idx]
                        if nei.size() == torch.Size([0]) or nej.size() == torch.Size([0]):
                            continue
                        sim_vals_nei = sim_matrix[batch_idx, nei]
                        S_hat[batch_idx,:len(nei),:len(nej)] += sim_vals_nei[:,nej]
                        S_hat_mask[batch_idx,:len(nei),:len(nej)] += 1

                    # to add: If sinkhorn normalisation fails to converge then warning 
                    S = masked_sinkhorn(S_hat, S_hat_mask.type(torch.bool), num_steps=num_steps, 
                                        device=self.device, tau=tau, dim=-1) # [B, max_i, max_j]
                    opt_perm = S.max(dim=-1).indices
                    max_mask = F.one_hot(opt_perm, num_classes = S.shape[-1]).type(torch.bool)
                    cost = torch.sum(S_hat.masked_fill(~max_mask, 0), (-1,-2))
                    
                    # Normalise update
                    M = torch.tensor([max(u,v) for u,v in zip(num_i, num_j)], device=self.device)
                    update = torch.div((cost-M*sim_update[:,i,j]), 1+M)

                    sim_update[:,i,j] += update

            sim_matrix[S_mask] += sim_update[S_mask]
            # sim_matrix = masked_softmax(sim_matrix, S_mask)
            # sim_matrix = torch.nn.functional.normalize(sim_matrix, dim=-1)
        
        sim_matrix = sim_matrix[s_mask] # [sum{N_s(i)}, N_t]
        sim_matrix = ((sim_matrix)).masked_fill(~sim_t_mask, 0)

        return sim_matrix

    def loss(self, S, gt, batch_s = None, reduction = "mean", regr=False):
        r'''
        Args:
            S (Tensor): Similarity matrix of size :obj:`[batchsum_{s}(N_{i}), N_t]`
            gt (Tensor): Ground truth mappings of size :obj:`[batchsum_{s}(N_{i}), 2]`
        '''
        if regr:
            score = scatter(S.max(dim=-1).values, batch_s)
            score = torch.mean(torch.square(score - gt))

            return score
        else:
            t_maps = gt.transpose(0,-1)[1]
            gt_map = torch.zeros(S.shape, device=self.device)
            batch_pairs = torch.stack((torch.arange(S.shape[0], device=self.device), t_maps), dim=1)
            for pair in batch_pairs:
                gt_map[pair[0], pair[1]] = 1

            val = scatter(torch.sum(S * gt_map, dim=-1), batch_s)
        nll = -torch.log(val+1e-8)
        
        return torch.mean(nll) if reduction == "mean" else torch.sum(nll) if reduction == "sum" else nll


    # def perm_loss(self, S, gt, batch = None, reduction = "mean"):
    #     r'''
    #     Args:
    #         S (Tensor): Similarity matrix of size :obj:`[batchsum_{s}(N_{i}), N_t]`
    #         gt (Tensor): Ground truth mappings of size :obj:`[batchsum_{s}(N_{i}), 2]`
    #     '''
    #     # Needs a faster implementation
    #     y = gt.transpose(0,-1)
    #     if batch is not None:
    #         num_classes = torch.bincount(batch)
    #         gt_map = to_dense_adj(y, batch) 
    #         gt_map = gt_map.view(gt_map.shape[0]*gt_map.shape[1], gt_map.shape[2])
    #         # gt_map = gt_map[gt_map.sum(dim=-1)>0] # perm matrix

    #         assert gt_map.shape[-1] == S.shape[-1]
    #         pred = self.predict(S, y[0])
    #         _max = torch.max(pred).values
    #         pred_perm = torch.zeros(gt_map.shape, device=self.device)
    #         idx = 0
    #         _prev = 0
    #         for i,j in enumerate(pred):
    #             if i > _prev+num_classes[idx]-1:
    #                 idx+=1
    #                 _prev += num_classes[idx]
    #             pred_perm[j,i-_prev] = 1
    #     else:
    #         val = S[y[0],y[1]].sum()
    #     perm_cel = -torch.sum(gt_map * torch.log(pred_perm+1e-8) + (1-gt_map)*torch.log(1-pred_perm+1e-8), dim=0)
    #     perm_cel.requires_grad_()
    #     return torch.mean(perm_cel) if reduction == "mean" else torch.sum(perm_cel) if reduction == "sum" else perm_cel

    def acc(self, S, gt, batch_s = None, reduction = "mean"):
        r'''
        Args:
            S (Tensor): Similarity matrix of size :obj:`[batchsum_{s}(N_{i}), N_t]`
            gt (Tensor): Ground truth mappings of size :obj:`[batchsum_{s}(N_{i}), 2]`
        '''
        # Include warning if batch not passed
        y = gt.transpose(0,-1)
        pred = self.predict(S, y[0])

        if batch_s is not None:
            t_maps = gt.transpose(0,-1)[1]
            mean_acc = torch.div(scatter((pred == t_maps).long(), batch_s), torch.bincount(batch_s))
        else:
            mean_acc = (pred == y[1]).sum().item() / y.shape[1]
        
        return torch.mean(mean_acc) if reduction == "mean" else mean_acc

    def p_at_k(self, S, gt, k, batch = None, reduction = "mean"):
        # Does not work yet
        r'''
        Args:
            S (Tensor): Similarity matrix of size :obj:`[batchsum_{s}(N_{i}), N_t]`
            gt (Tensor): Ground truth mappings of size :obj:`[batchsum_{s}(N_{i}), 2]`
        '''
        # Include warning if batch not passed
        y = gt.transpose(0,-1)
        pred = self.predict(S, y[0])

        if batch is not None:
            t_maps = gt.transpose(0,-1)[1]
            counts = torch.bincount(batch)
            for idx, c in enumerate(counts):
                if k>c:
                    mean_acc = torch.div(scatter((pred == t_maps).long(), batch), torch.bincount(batch))
            else:
                mean_acc = torch.div(scatter((pred[:k] == t_maps[:k]).long(), batch), torch.bincount(batch))
        else:
            mean_acc = (pred == y[1]).sum().item() / y.shape[1]
        
        return torch.mean(mean_acc) if reduction == "mean" else mean_acc
    
    def predict(self, S, r):
        r = torch.arange(len(r))
        return S[r].argmax(dim=-1)
