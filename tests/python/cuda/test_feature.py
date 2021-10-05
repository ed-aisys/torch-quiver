import torch
import torch_quiver as torch_qv
import random
import numpy as np
import time
from typing import List
from quiver.shard_tensor import ShardTensor, ShardTensorConfig, Topo

class Feature:
    def __init__(self, rank, device_list, device_cache_size=0, cache_policy='device_replicate'):
        self.device_cache_size = device_cache_size
        self.cache_policy = cache_policy
        self.device_list = device_list
        self.device_tensor_list = {}
        self.numa_tensor_list = dict.fromkeys([0, 1], None)
        self.rank = rank            
        self.topo = Topo(self.device_list)
    
    def cal_memory_budget_bytes(self, memory_budget):
        if isinstance(memory_budget, int):
            return memory_budget
        elif isinstance(memory_budget, float):
            memory_budget = int(memory_budget)
        elif isinstance(memory_budget, str):
            if memory_budget.upper().endswith("M") or memory_budget.upper().endswith("MB"):
                end = -1 if memory_budget.upper().endswith("M") else -2
                memory_budget = int(float(memory_budget[:end]) * 1024 * 1024)
            elif memory_budget.upper().endswith("G") or memory_budget.upper().endswith("GB"):
                end = -1 if memory_budget.upper().endswith("G") else -2
                memory_budget = int(float(memory_budget[:end]) * 1024 * 1024 * 1024)
        else:
            raise Exception("memory budget input is not valid")
        return memory_budget


    def cal_size(self, cpu_tensor, cache_memory_budget):
        element_size = cpu_tensor.shape[1] * 4
        cache_size = cache_memory_budget // element_size
        return cache_size

    def partition(self, cpu_tensor, cache_memory_budget):
        
        cache_size = self.cal_size(cpu_tensor, cache_memory_budget)
        return [cpu_tensor[:cache_size], cpu_tensor[cache_size: ]]

    def from_cpu_tensor(self, cpu_tensor):
        if self.cache_policy == "device_replicate":
            cache_memory_budget = self.cal_memory_budget_bytes(self.device_cache_size)
        else:
            cache_memory_budget = self.cal_memory_budget_bytes(self.device_cache_size) * len(self.topo.Numa2Device[0])
        
        print(f"LOG>>> {min(100, int(100 * cache_memory_budget / cpu_tensor.numel() / 4))}% data cached")
        cache_part, self.cpu_part = self.partition(cpu_tensor, cache_memory_budget)
        if self.cache_policy == "device_replicate":
            for device in self.device_list:
                shard_tensor = ShardTensor(self.rank, ShardTensorConfig({}))
                shard_tensor.append(cache_part, device)
                self.device_tensor_list[device] = shard_tensor
            if self.cpu_part.numel() > 0:
                self.device_tensor_list[self.rank].append(self.cpu_part, -1)

        else:
            numa0_device_list = self.topo.Numa2Device[0]
            numa1_device_list = self.topo.Numa2Device[1]
            block_size = self.cal_size(cpu_tensor, cache_memory_budget // len(self.topo.Numa2Device[0]))

            if len(numa0_device_list) > 0:
                print(f"LOG>>> GPU {numa0_device_list} belong to the same NUMA Domain")
                shard_tensor = ShardTensor(self.rank, ShardTensorConfig({}))
                cur_pos = 0
                for device in numa0_device_list:
                    shard_tensor.append(cache_part[cur_pos: cur_pos + block_size], device)
                    cur_pos += block_size
                    if cur_pos > block_size * len(self.topo.Numa2Device[0]):
                        break

                self.numa_tensor_list[0] = shard_tensor
            
            if len(numa1_device_list) > 0:
                print(f"LOG>>> GPU {numa1_device_list} belong to the same NUMA Domain")
                shard_tensor = ShardTensor(self.rank, ShardTensorConfig({}))
                cur_pos = 0
                for device in numa1_device_list:
                    shard_tensor.append(cache_part[cur_pos: cur_pos + block_size], device)
                    cur_pos += block_size

                self.numa_tensor_list[1] = shard_tensor

            if self.cpu_part.numel() > 0:
                numa_id = self.topo.get_numa_node(self.rank)
                self.numa_tensor_list[numa_id].append(self.cpu_part, -1)

        
    def __getitem__(self, node_idx):
        node_idx = node_idx.to(self.rank)
        if self.cache_policy == "device_replicate":
            shard_tensor = self.device_tensor_list[self.rank]
            return shard_tensor[node_idx]
        else:
            numa_id = self.topo.get_numa_node(self.rank)
            shard_tensor = self.numa_tensor_list[numa_id]
            return shard_tensor[node_idx]


def test_feature_basic():
    rank = 0
    
    NUM_ELEMENT = 100000
    SAMPLE_SIZE = 8000
    FEATURE_DIM = 600
    #########################
    # Init With Numpy
    ########################
    torch.cuda.set_device(rank)
    torch_qv.init_p2p()


    host_tensor = np.random.randint(
        0, high=10, size=(2 * NUM_ELEMENT, FEATURE_DIM))
    
    print("host data size", host_tensor.size * 4 // 1024  // 1024, "MB")
    tensor = torch.from_numpy(host_tensor).type(torch.float32)

    host_indice = np.random.randint(0, 2 * NUM_ELEMENT - 1, (SAMPLE_SIZE, ))
    indices = torch.from_numpy(host_indice).type(torch.long)

    device_indices = indices.to(rank)

    #########################
    # define a quiver.Feature
    #########################
    feature = Feature(rank=0, device_list=[0, 1], device_cache_size="0.2G", cache_policy="numa_replicate")
    feature.from_cpu_tensor(tensor)

    ####################
    # Indexing 
    ####################
    res = feature[device_indices]
    
    start = time.time()

    res = feature[device_indices]
    consumed_time = time.time() - start



    res = res.cpu().numpy()
    feature_gt = tensor[indices].numpy()
    print("Correctness Check : ", np.array_equal(res, feature_gt))
    print(
        f"TEST SUCCEED!, With Memory Bandwidth = {res.size * 4 / consumed_time / 1024 / 1024 / 1024} GB/s, consumed {consumed_time}s")
    while 1:
        pass

test_feature_basic()