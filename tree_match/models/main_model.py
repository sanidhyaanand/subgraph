from turtle import update
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_scatter import scatter

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
    def __init__(self, psi_1, k=-1):
        super(MainModel, self).__init__()
        self.psi_1 = psi_1

    def reset_parameters(self):
        self.psi_1.reset_parameters()
    
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x_s, edge_index_s, batch_s,
                x_t, edge_index_t, batch_t, include_gnn = True,
                num_steps = 5, tau = 0.1):
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
        
        # Convert [sum(N_{i}), C_out] tensor to [B = batch_size, N = max(N_{i}) where i in range(B), C_out] tensor
        h_s, s_mask = to_dense_batch(h_s, batch_s, fill_value=0) # [B, N_s = max_{s}(N_{i}), C_out], [B, N_s]
        h_t, t_mask = to_dense_batch(h_t, batch_t, fill_value=0) # [B, N_t = max_{t}(N_{i}), C_out], [B, N_t]
        adj_s = to_dense_adj(edge_index=edge_index_s, batch=batch_s)
        adj_t = to_dense_adj(edge_index=edge_index_t, batch=batch_t)

        sim_matrix = h_s @ h_t.transpose(-1,-2) # [B, N_s, N_t]
        sim_update = torch.zeros(sim_matrix.shape, device=self.device)
        (B, N_s, N_t) = sim_matrix.size()
        S_mask = s_mask.view(B, N_s, 1) & t_mask.view(B, 1, N_t)
        sim_t_mask = t_mask.repeat_interleave(torch.bincount(batch_t), dim=0)

        # Somehow tensorize this through masking?
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

                # add: If sinkhorn normalisation fails to converge then warning 
                S = masked_sinkhorn(S_hat, S_hat_mask.type(torch.bool), num_steps=num_steps, 
                                    device=self.device, tau=tau, dim=-1) # [B, max_i, max_j]
                opt_perm = S.max(dim=-1).indices
                max_mask = F.one_hot(opt_perm, num_classes = S.shape[-1]).type(torch.bool)
                cost = torch.sum(S_hat.masked_fill(~max_mask, 0), (-1,-2))
                M = torch.tensor([max(u,v) for u,v in zip(num_i, num_j)], device=self.device)
                update = torch.div((cost-M*sim_update[:,i,j]), 1+M)

                # update needs to be normalised to not disturb scaling
                sim_update[:,i,j] += update
        
        sim_matrix[S_mask] += sim_update[S_mask]
        sim_matrix = masked_softmax(sim_matrix, S_mask)
        
        return sim_matrix[s_mask].masked_fill(~sim_t_mask, 0) # [sum_{s}(N_{i}), N_t]

    
    def loss(self, S, gt, batch = None):
        r'''
        Args:
            S (Tensor): Similarity matrix of size :obj:`[batchsum_{s}(N_{i}), N_t]`
            gt (Tensor): Ground truth mappings of size :obj:`[batchsum_{s}(N_{i}), 2]`
        '''
        y = gt.transpose(0,-1)
        if batch is not None:
            gt_map = to_dense_adj(y, batch)
            gt_map = gt_map.view(gt_map.shape[0]*gt_map.shape[1], gt_map.shape[2])
            gt_map = gt_map[gt_map.sum(dim=-1)>0]

            assert gt_map.shape[-1] == S.shape[-1]
            val = scatter(torch.sum(S * gt_map, dim=-1), batch)
        else:
            val = S[y[0],y[1]].sum()
        nll = -torch.log(val+1e-8)
        return nll

    def acc(self, S, gt, batch = None):
        r'''
        Args:
            S (Tensor): Similarity matrix of size :obj:`[batchsum_{s}(N_{i}), N_t]`
            gt (Tensor): Ground truth mappings of size :obj:`[batchsum_{s}(N_{i}), 2]`
        '''
        # Include warning if batch not passed
        y = gt.transpose(0,-1)
        pred = self.predict(S, y[0])
        if batch is not None:
            gt_map = to_dense_adj(y, batch)
            gt_map = gt_map.view(gt_map.shape[0]*gt_map.shape[1], gt_map.shape[2])
            gt_map = gt_map[gt_map.sum(dim=-1)>0].argmax(dim=-1)
            mean_acc = scatter(pred == gt_map, batch) / y.shape[1]
        else:
            mean_acc = (pred == y[1]).sum().item() / y.shape[1]
        
        return mean_acc
    
    def predict(self, S, r):
        return S[r].argmax(dim=-1)
