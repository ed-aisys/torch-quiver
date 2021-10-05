
import torch_quiver as torch_qv
import torch
import random
from typing import List
from .shard_tensor import ShardTensor, ShardTensorConfig, Topo

class Feature:
    def __init__(self, rank, device_cache_size, device_list = [], cache_policy='auto'):
        self.device_cache_size = device_cache_size
        self.cache_policy = cache_policy
        self.device_list = device_list
        self.device_tensor_list = {}
        self.numa_tensor_list = dict.from_keys([0, 1], None)
        self.rank = rank
        self.topo = Topo(self.device_list)

    def from_cpu_tensor(self, cpu_tensor):
        if self.cache_policy == "device_replicate":
            for device in self.device_list:
                shard_tensor_config = ShardTensorConfig({device: self.device_cache_size})
                shard_tensor = ShardTensor(self.rank, shard_tensor_config)
                shard_tensor.from_cpu_tensor(cpu_tensor)
                self.device_tensor_list[device] = shard_tensor
        else:
           
            numa0_device_list = self.topo.Numa2Device[0]
            numa1_device_list = self.topo.Numa2Device[1]
            if len(numa0_device_list) > 0:
                print(f"Quiver LOG>>> GPU {numa0_device_list} belong to the same NUMA Domain")
                shard_tensor_config = ShardTensorConfig(dict.from_keys(numa0_device_list, self.device_cache_size))
                shard_tensor = ShardTensor(self.rank, shard_tensor_config)
                shard_tensor.from_cpu_tensor(cpu_tensor)
                self.numa_tensor_list[0] = shard_tensor
            
            if len(numa1_device_list) > 0:
                print(f"Quiver LOG>>> GPU {numa1_device_list} belong to the same NUMA Domain")
                shard_tensor_config = ShardTensorConfig(dict.from_keys(numa1_device_list, self.device_cache_size))
                shard_tensor = ShardTensor(self.rank, shard_tensor_config)
                shard_tensor.from_cpu_tensor(cpu_tensor)
                self.numa_tensor_list[1] = shard_tensor
            

    def __getitem__(self, node_idx):
        node_idx = node_idx.to(self.rank)
        if self.cache_policy == "device_replicate":
            shard_tensor = self.device_tensor_list[self.rank]
            return shard_tensor[node_idx]
        else:
            numa_id = self.topo.get_numa_node(self.rank)
            shard_tensor = self.numa_tensor_list[numa_id]
            return shard_tensor[node_idx]
        
    
    def share_ipc(self):
        if self.cache_policy == "device_replicate":
            pass
        
        







            