"""
TUDataset wrapper with Yehudai et al. (ICML 2021) size-based splits.

Split protocol:
  1. Sort all graphs by num_nodes
  2. Bottom 50% → pool_small, Top 10% → test, Middle 40% → discarded
  3. Within pool_small: stratified 90% train / 10% val (seeded by split_seed)

Uses preprocess_item_fast (BFS-based SPD, dummy edge features).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.datasets import TUDataset

from graphormer.data.wrapper import preprocess_item_fast


class TUDatasetSizeGen(Dataset):
    def __init__(self, name, split="train", root="dataset", cache_dir="datasets/tudataset",
                 split_seed=0):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        cache_path = os.path.join(cache_dir, f"{name.lower()}.pt")

        if os.path.exists(cache_path):
            print(f"Loading cached {name} from {cache_path}")
            cached = torch.load(cache_path, map_location="cpu")
            all_data = cached["data"]
            splits = cached["splits"]
        else:
            print(f"Preprocessing {name} (one-time)...")
            pyg_dataset = TUDataset(root=root, name=name, use_node_attr=False)

            all_data = []
            for i, item in enumerate(pyg_dataset):
                # Normalise x to [N, 1] LongTensor
                x = item.x
                if x is None:
                    # No node features — use zeros
                    x = torch.zeros(item.num_nodes, 1, dtype=torch.long)
                elif x.dim() == 2 and x.size(1) > 1:
                    # One-hot (e.g. NCI1: [N, 37]) → integer label
                    x = x.argmax(dim=1, keepdim=True).long()
                else:
                    x = x.long()
                if x.dim() == 1:
                    x = x.unsqueeze(1)
                item.x = x

                # Ensure edge_attr exists as [E, 1] zeros
                E = item.edge_index.size(1) if item.edge_index is not None else 0
                item.edge_attr = torch.zeros(E, 1, dtype=torch.long)

                processed = preprocess_item_fast(item)
                # Clamp spatial_pos: disconnected components produce inf → large int
                sp = processed.spatial_pos
                sp[sp < 0] = 510  # overflow from inf cast
                sp.clamp_(max=510)
                all_data.append({
                    "x": processed.x,
                    "edge_index": processed.edge_index if hasattr(processed, 'edge_index') else item.edge_index,
                    "edge_attr": item.edge_attr,
                    "y": item.y.view(-1).long(),
                    "num_nodes": processed.x.size(0),
                    "attn_bias": processed.attn_bias,
                    "attn_edge_type": processed.attn_edge_type,
                    "spatial_pos": sp,
                    "in_degree": processed.in_degree,
                    "out_degree": processed.out_degree,
                    "edge_input": processed.edge_input,
                })
                if (i + 1) % 200 == 0:
                    print(f"  {i+1}/{len(pyg_dataset)} graphs processed")

            # Build size-based splits
            sizes = np.array([d["num_nodes"] for d in all_data])
            labels = np.array([d["y"].item() for d in all_data])
            order = np.argsort(sizes, kind="stable")

            n_total = len(all_data)
            n_small = n_total // 2          # bottom 50%
            n_test = n_total // 10          # top 10%

            small_idx = order[:n_small]
            test_idx = order[n_total - n_test:]

            # Stratified 90/10 train/val within small pool
            rng = np.random.RandomState(split_seed)
            small_labels = labels[small_idx]
            train_idx_list, val_idx_list = [], []
            for cls in np.unique(small_labels):
                cls_mask = small_labels == cls
                cls_indices = small_idx[cls_mask]
                rng.shuffle(cls_indices)
                n_val = max(1, len(cls_indices) // 10)
                val_idx_list.append(cls_indices[:n_val])
                train_idx_list.append(cls_indices[n_val:])
            train_idx = np.concatenate(train_idx_list)
            val_idx = np.concatenate(val_idx_list)

            splits = {
                "train": train_idx.tolist(),
                "val": val_idx.tolist(),
                "test": test_idx.tolist(),
            }

            os.makedirs(cache_dir, exist_ok=True)
            torch.save({"data": all_data, "splits": splits}, cache_path)
            print(f"Saved cache to {cache_path} ({len(all_data)} graphs, "
                  f"train={len(splits['train'])}, val={len(splits['val'])}, "
                  f"test={len(splits['test'])})")

        self.indices = splits[split]
        self.data = all_data
        print(f"{name} {split}: {len(self.indices)} graphs")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.data[self.indices[idx]]
