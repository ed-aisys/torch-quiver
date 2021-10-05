import torch
import numpy as np 
import scipy.sparse as sp
import torch_quiver as qv

import time
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from scipy.sparse import csr_matrix
import os
import os.path as osp

from quiver.sage_sampler import GraphSageSampler


def test_GraphSageSampler():
    """
    class GraphSageSampler:

    def __init__(self, edge_index: Union[Tensor, SparseTensor], sizes: List[int], device, num_nodes: Optional[int] = None, mode="UVA", device_replicate=True):
    """
    print(f"{'*' * 10} TEST WITH REAL GRAPH {'*' * 10}")

    root="/home/zy/products/"
    dataset = PygNodePropPredDataset('ogbn-products', root)
    data = dataset[0]
    edge_index = data.edge_index
    seeds_size = 128
    neighbor_size = 5
    
    seeds = np.arange(2000000)
    np.random.shuffle(seeds)
    seeds =seeds[:seeds_size]
    seeds = torch.from_numpy(seeds).type(torch.long)
    seeds = seeds.to(0)
    print("LOG>>> Create GraphSampler")
    ###########################
    # Create GraphSageSampler
    ###########################
    sage_sampler = GraphSageSampler(data.edge_index, sizes=[15, 10, 5], device=0)
    ###########################
    # Sample 
    ###########################
    res = sage_sampler.sample(seeds)

    start = time.time()
    res = sage_sampler.sample(seeds)
    print(f"LOG>>> Sample Finished, {time.time() - start}s consumed")
    

test_GraphSageSampler()
