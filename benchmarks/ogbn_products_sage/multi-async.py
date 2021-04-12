import argparse
import multiprocessing as mp

import torch
import torch.nn.functional as F
from torch.distributed import rpc

import os
import os.path as osp
import queue
import time

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.nn import SAGEConv
from quiver.cuda_sampler import CudaNeighborSampler
from torch_geometric.data import NeighborSampler

import horovod.torch as hvd

p = argparse.ArgumentParser(description='')
# p.add_argument('--node', type=int, default=0)
# p.add_argument('--num_nodes', type=int, default=1)
p.add_argument('--gpu', type=int, default=4)
p.add_argument('--batch_size', type=int, default=512)
p.add_argument('--ip', type=str, default='localhost')
args = p.parse_args()


class ProductsDataset:
    def __init__(self, train_idx, edge_index, x, y, f, c):
        self.train_idx = train_idx
        self.edge_index = edge_index
        self.x = x
        self.y = y
        self.num_features = f
        self.num_classes = c


class SyncConfig:
    def __init__(self, ip, group, rank, world_size, queues, barriers, timeout,
                 reduce, dist):
        self.ip = ip
        self.group = group
        self.rank = rank
        self.world_size = world_size
        self.queues = queues
        self.barriers = barriers
        self.timeout = timeout
        self.reduce = reduce
        self.dist = dist


local_rank = None
loader = None
selector = None
sample_queue = None
num_device = 4


def sample_cuda(nodes, size):
    torch.cuda.set_device(local_rank)
    nodes = loader.global2local(nodes)
    neighbors, counts = loader.quiver.sample_neighbor(0, nodes, size)
    neighbors = loader.local2global(neighbors)

    return neighbors, counts


def sample_cpu(nodes, size):
    nodes = loader.global2local(nodes)
    neighbors, counts = loader.quiver.sample_neighbor(nodes, size)
    neighbors = loader.local2global(neighbors)

    return neighbors, counts


def node_f(nodes, is_feature):
    if is_feature:
        return selector.x[nodes]
    else:
        return selector.y[nodes]


def select_f(nodes, is_feature):
    return selector.get_data(nodes, is_feature)


def get_f(rank):
    res = None
    try:
        res = sample_queue.get(timeout=10)
    except queue.Empty:
        print('timeout')
    return res


get_sample = get_f
select_features = select_f


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(SAGE, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)


# def FeatureProcess(sync, x, y, f_rank, f_size):
#     global selector
#     torch.set_num_threads(5)

#     import quiver.dist_cpu_sampler as dist
#     comm = dist.Comm(f_rank, f_size)

#     def node2rank(nodes):
#         ranks = torch.fmod(nodes, f_size)
#         return ranks

#     def local2global(nodes):
#         return nodes

#     def global2local(nodes):
#         return nodes

#     selector = dist.SyncDistFeatureSelector(
#         comm, ((x, y), local2global, global2local, node2rank))
#     dist.node_features = node_f
#     dist.select_features = select_f
#     port = 27777
#     os.environ['MASTER_ADDR'] = sync.ip
#     os.environ['MASTER_PORT'] = str(port)
#     rpc.init_rpc(f"feature{f_rank}",
#                  rank=sync.rank,
#                  world_size=sync.world_size,
#                  rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
#                      num_worker_threads=1, rpc_timeout=20))
#     sync.barriers[0].wait()
#     rpc.shutdown()
#     time.sleep(30)


def SampleProcess(sync, dev, edge_index, train_idx, sizes, batch_size, s_rank,
                  s_size):
    global loader
    global local_rank
    device, num = dev
    group = sync.group
    torch.set_num_threads(5)

    def node2rank(nodes):
        ranks = torch.fmod(nodes, s_size)
        return ranks

    def local2global(nodes):
        return nodes

    def global2local(nodes):
        return nodes

    if device != 'cpu':
        torch.cuda.set_device(device)
        if sync.dist:
            import quiver.dist_cuda_sampler as dist
            comm = dist.Comm(s_rank, s_size)
            local_rank = device
            loader = dist.SyncDistNeighborSampler(
                comm, (int(edge_index.max() + 1), edge_index,
                       (None, None), local2global, global2local, node2rank),
                train_idx, [15, 10, 5],
                device,
                node_f,
                batch_size=batch_size)
            dist.sample_neighbor = sample_cuda
        else:
            loader = CudaNeighborSampler(edge_index,
                                         device=device,
                                         rank=num,
                                         node_idx=train_idx,
                                         sizes=sizes,
                                         batch_size=batch_size,
                                         shuffle=True)
    else:
        if sync.dist:
            import quiver.dist_cpu_sampler as dist
            comm = dist.Comm(s_rank, s_size)
            loader = dist.SyncDistNeighborSampler(
                comm, (int(edge_index.max() + 1), edge_index,
                       torch.zeros(1, dtype=torch.long), local2global,
                       global2local, node2rank),
                train_idx, [15, 10, 5],
                sync.rank,
                batch_size=batch_size)
            dist.sample_neighbor = sample_cpu
        else:
            loader = NeighborSampler(edge_index,
                                     node_idx=train_idx,
                                     sizes=sizes,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=num)
    if sync.dist:
        os.environ['MASTER_ADDR'] = sync.ip
        port = 27777
        os.environ['MASTER_PORT'] = str(port)
        rpc.init_rpc(f"worker{s_rank}",
                     rank=sync.rank,
                     world_size=sync.world_size,
                     rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                         num_worker_threads=1, rpc_timeout=20))
    cont = True
    count = 0
    wait = 0.0
    while cont:
        for sample in loader:
            count += 1
            if count == 1:
                sync.barriers[0].wait()
            try:
                t0 = time.time()
                if count > 1:
                    sync.queues[0].put((device, sample), timeout=sync.timeout)
                wait += time.time() - t0
            except queue.Full:
                cont = False
                break
    rpc.shutdown()
    time.sleep(30)


def SampleProxy(sync, p_rank=0, p_size=1):
    os.environ['MASTER_ADDR'] = sync.ip
    port = 27777
    os.environ['MASTER_PORT'] = str(port)
    global sample_queue
    sample_queue = sync.queues[0]
    rpc.init_rpc(f"proxy{p_rank}",
                 rank=sync.rank,
                 world_size=sync.world_size,
                 rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                     num_worker_threads=1, rpc_timeout=20))
    sync.barriers[0].wait()
    rpc.shutdown()
    time.sleep(30)


def TrainProcess(ip, num_batch, g_size, batch_size):
    torch.set_num_threads(5)
    import quiver.dist_cpu_sampler as dist
    hvd.init()
    rank = hvd.rank()
    ws = hvd.size() + g_size + 1
    tmp_rank = hvd.size()
    device = torch.device(hvd.local_rank())
    root = './products.pt'
    dataset = torch.load(root)
    train_idx = dataset.train_idx
    edge_index = dataset.edge_index
    x, y = dataset.x.share_memory_(), dataset.y.squeeze().share_memory_()
    model = SAGE(dataset.num_features, 256, dataset.num_classes, 3)
    model = model.to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    optimizer = hvd.DistributedOptimizer(optimizer)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    if hvd.local_rank() == 0:
        queues = [mp.Queue(64)]
        barrier_num = g_size + 2
        barriers = [mp.Barrier(barrier_num)]
        samplers = []
        tmp_dev = num_device - 1
        for i in range(g_size):
            sync = SyncConfig(ip, 0, tmp_rank, ws, queues, barriers, 3, True,
                              True)
            sampler = mp.Process(target=SampleProcess,
                                 args=(sync, (tmp_dev, 0), edge_index,
                                       train_idx, [15, 10, 5],
                                       batch_size // hvd.size(), i, g_size))
            sampler.start()
            samplers.append(sampler)
            tmp_rank += 1
            tmp_dev -= 1
        # if c_rank >= 0:
        #     sync = SyncConfig(ip, 0, tmp_rank, ws, queues, barriers, 3, True,
        #                       True)
        #     sampler = mp.Process(target=SampleProcess,
        #                          args=(sync, ('cpu', 5), data.edge_index,
        #                                train_idx, [15, 10, 5],
        #                                512 // hvd.size(), c_rank, c_size))
        #     sampler.start()
        #     samplers.append(sampler)
        #     tmp_rank += 1
        sync = SyncConfig(ip, 0, tmp_rank, ws, queues, barriers, 3, True, True)
        sampler_proxy = mp.Process(target=SampleProxy,
                                   args=(sync,))
        sampler_proxy.start()
        tmp_rank += 1
        # sync = SyncConfig(ip, 0, tmp_rank, ws, queues, barriers, 3, True, True)
        # feature_selector = mp.Process(target=FeatureProcess,
        #                               args=(sync, data.x, data.y.squeeze(),
        #                                     f_rank, f_size))
        # feature_selector.start()
        # tmp_rank += 1
    port = 27777
    os.environ['MASTER_ADDR'] = ip
    os.environ['MASTER_PORT'] = str(port)
    rpc.init_rpc(
        f"train{rank}",
        rank=rank,  # TODO: distributed
        world_size=ws,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=1, rpc_timeout=20))
    if rank == 0:
        barriers[0].wait()
    print(f'{rank} ready')
    hvd.allreduce(torch.tensor(0), name='barrier')
    t0 = time.time()
    sample_time = 0.0
    feature_time = 0.0
    train_time = 0.0
    for i in range(num_batch):
        beg = time.time()
        sample = rpc.rpc_sync(
            f"proxy0",  # TODO: distributed
            get_sample,
            args=(0, ),
            kwargs=None,
            timeout=-1.0)
        _, sample = sample
        batch_size, n_id, adjs = sample
        sample_time += time.time() - beg
        beg = time.time()
        feature = x[n_id].to(device)
        label = y[n_id[:batch_size]].to(device)
        # feature = rpc.rpc_sync(
        #     f"feature{f_rank}",  # TODO: distributed
        #     select_features,
        #     args=(n_id, True),
        #     kwargs=None,
        #     timeout=-1.0)
        # label = rpc.rpc_sync(
        #     f"feature{f_rank}",  # TODO: distributed
        #     select_features,
        #     args=(n_id[:batch_size], False),
        #     kwargs=None,
        #     timeout=-1.0)
        feature_time += time.time() - beg
        beg = time.time()
        adjs = [adj.to(device) for adj in adjs]
        # feature, order0 = feature
        # label, order1 = label
        # feature = torch.cat([f.to(device) for f in feature])
        # label = torch.cat([l.to(device) for l in label])
        # origin_feature = torch.empty_like(feature)
        # origin_label = torch.empty_like(label)
        # origin_feature[order0] = feature
        # origin_label[order1] = label

        optimizer.zero_grad()
        out = model(feature, adjs)
        loss = F.nll_loss(out, label)
        loss.backward()
        optimizer.step()
        train_time += time.time() - beg
    hvd.allreduce(torch.tensor(0), name='barrier')
    t = time.time() - t0
    if rank == 0:
        print(f'took {t}')
        print(f'sample {sample_time}')
        print(f'feature {feature_time}')
        print(f'train {train_time}')
    rpc.shutdown()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    batch_size = args.batch_size
    g_size = args.gpu

    TrainProcess(args.ip, 192, g_size, batch_size)
