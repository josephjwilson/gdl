import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
import networkx as nx
from graphormer.data.wrapper import preprocess_item, preprocess_item_fast
from ogb.graphproppred import PygGraphPropPredDataset

class PurePositionalDataset(Dataset):
    NUM_CLASSES = 1

    def __init__(
        self,
        num_samples: int = 2000,
        num_nodes: int = 12,
        n_nodes: int = None,
        max_distance: int = 12,
        feature_dim: int = None,
        seed: int = 42,
        cache_path: str = None,
    ):
        super().__init__()
        if n_nodes is not None:
            num_nodes = n_nodes
        self.data_list = []
        self.max_distance = max_distance
        rng = np.random.default_rng(seed)

        if cache_path and os.path.exists(cache_path):
            cached = torch.load(cache_path, map_location='cpu')
            self.data_list = cached['data_list']
            print(f"Loaded {len(self.data_list)} positional graphs from {cache_path}")
        else:
            for _ in range(num_samples):
                edge_index_list = self._generate_random_tree(rng, num_nodes)
                edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()

                avg_path_length = self._compute_avg_path_length(edge_index_list, num_nodes)

                x = torch.full((num_nodes, 1), 10, dtype=torch.long)

                data = Data(
                    x=x,
                    edge_index=edge_index,
                    y=torch.tensor([avg_path_length], dtype=torch.float),
                    num_nodes=num_nodes,
                )

                data.edge_attr = torch.zeros((edge_index.size(1), 1), dtype=torch.long)
                data = preprocess_item(data)
                self.data_list.append(data)
            if cache_path:
                os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
                torch.save({'data_list': self.data_list, 'num_nodes': num_nodes, 'seed': seed}, cache_path)
                print(f"Saved cache to {cache_path}")

    @staticmethod
    def _generate_random_tree(rng, num_nodes):
        import heapq

        if num_nodes == 1:
            return []
        if num_nodes == 2:
            return [[0, 1], [1, 0]]

        prufer = list(rng.integers(0, num_nodes, size=num_nodes - 2))
        degree = [1] * num_nodes
        for node in prufer:
            degree[node] += 1

        leaves = [i for i in range(num_nodes) if degree[i] == 1]
        heapq.heapify(leaves)

        edges = []
        for node in prufer:
            leaf = heapq.heappop(leaves)
            edges.append([leaf, node])
            edges.append([node, leaf])
            degree[leaf] -= 1
            degree[node] -= 1
            if degree[node] == 1:
                heapq.heappush(leaves, node)

        u = heapq.heappop(leaves)
        v = heapq.heappop(leaves)
        edges.append([u, v])
        edges.append([v, u])
        return edges

    @staticmethod
    def _compute_diameter(edge_index_list, num_nodes):
        adj = [[] for _ in range(num_nodes)]
        seen = set()
        for u, v in edge_index_list:
            if (u, v) not in seen:
                adj[u].append(v)
                seen.add((u, v))

        def bfs(start):
            dist = [-1] * num_nodes
            dist[start] = 0
            queue = [start]
            idx = 0
            while idx < len(queue):
                node = queue[idx]; idx += 1
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        queue.append(nb)
            farthest = max(range(num_nodes), key=lambda i: dist[i] if dist[i] >= 0 else -1)
            return farthest, dist[farthest]

        far1, _ = bfs(0)
        _, diam = bfs(far1)
        return diam

    @staticmethod
    def _compute_avg_path_length(edge_index_list, num_nodes):
        adj = [[] for _ in range(num_nodes)]
        seen = set()
        for u, v in edge_index_list:
            if (u, v) not in seen:
                adj[u].append(v)
                seen.add((u, v))

        total = 0
        for start in range(num_nodes):
            dist = [-1] * num_nodes
            dist[start] = 0
            queue = [start]
            idx = 0
            while idx < len(queue):
                node = queue[idx]; idx += 1
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        queue.append(nb)
            for j in range(start + 1, num_nodes):
                total += dist[j]

        num_pairs = num_nodes * (num_nodes - 1) // 2
        return total / num_pairs if num_pairs > 0 else 0.0

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            subset = PurePositionalDataset.__new__(PurePositionalDataset)
            subset.data_list = self.data_list[idx]
            subset.max_distance = self.max_distance
            return subset

        data = self.data_list[idx]

        # Data output
        return {
            "x": data.x,
            "edge_index": data.edge_index,
            "edge_attr": data.edge_attr,
            "y": data.y,
            "num_nodes": data.num_nodes,
            "attn_bias": data.attn_bias,
            "attn_edge_type": data.attn_edge_type,
            "spatial_pos": data.spatial_pos,
            "in_degree": data.in_degree,
            "out_degree": data.out_degree,
            "edge_input": data.edge_input,
        }
