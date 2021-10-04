import asyncio
import concurrent
import copy
import os
import time
import numpy as np

from torch_sparse import SparseTensor
import torch
import torch_quiver as qv


class GraphSageSampler:

    def __init__(self, edge_index: Union[Tensor, SparseTensor], sizes: List[int], device, num_nodes: Optional[int] = None, mode="UVA", device_replicate=True):
        edge_index = edge_index.to("cpu")
        self.is_sparse_tensor = isinstance(edge_index, SparseTensor)

        self.sizes = sizes
        
        self.quiver = None
           # Obtain a *transposed* `SparseTensor` instance.
        if not self.is_sparse_tensor:
            if num_nodes is None:
                num_nodes = int(edge_index.max()) + 1

            value = None
            self.adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], value=value, sparse_sizes=(num_nodes, num_nodes)).t()
        else:
            self.adj_t = edge_index

        self.mode = mode
        if self.mode == "UVA":
            indptr, indices, _ = self.adj_t.csr()
            edge_id = torch.zeros(1, dtype=torch.long)
            self.quiver = qv.new_quiver_from_csr_array(indptr, indices, edge_id, device, device_replicate)
        else:
            pass
        
        self.device_replicate = device_replicate
        self.device = device

    
    def sample_layer(self, batch, size):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)
        n_id = batch.to(torch.device(self.device))
        n_id, count = self.quiver.sample_neighbor(0, n_id, size)
        return n_id, count
    
    def reindex(self, inputs, outputs, counts):
        return qv.reindex_single(inputs, outputs, counts)

    def sample(self, input_nodes):
        nodes = input_nodes.to(self.device)
        adjs = []

        batch_size = len(nodes)
        for size in self.sizes:
            out, cnt = self.sample_layer(nodes, size)
            frontier, row_idx, col_idx = self.reindex(nodes, out, cnt)
            row_idx, col_idx = col_idx, row_idx
            edge_index = torch.stack([row_idx, col_idx], dim=0)

            adj_size = torch.LongTensor([
                frontier.size(0),
                nodes.size(0),
            ])
            e_id = torch.tensor([])
            adjs.append(Adj(edge_index, e_id, adj_size))
            nodes = frontier

        return nodes, batch_size, adjs[::-1]

