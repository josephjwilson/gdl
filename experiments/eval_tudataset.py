# Evaluate trained TUDataset checkpoints following the size-generalisation
# protocol of Yehudai et al. (ICML 2021): small graphs for train/val,
# large graphs held out for test. Gap between val and test acc is the metric.

import os
import sys
import re
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from types import SimpleNamespace

_this_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(_this_dir)
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "fairseq"))

from graphormer.models.graphormer import GraphormerModel, GraphormerEncoder
from graphormer.data.collator import collator
from config import ModelConfig
from data.tu_dataset import TUDatasetSizeGen


class TaskHead(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(dim, out_dim)

    def forward(self, x):
        return self.proj(x[:, 0, :])  # CLS readout


def make_collator(max_nodes):
    def fn(items):
        objs = []
        for i, item in enumerate(items):
            obj = SimpleNamespace(**item)
            obj.idx = i
            objs.append(obj)
        valid = [o for o in objs if o.x.size(0) <= max_nodes]
        if not valid:
            return {}
        return collator(valid, max_node=max_nodes, multi_hop_max_dist=5, spatial_pos_max=512)
    return fn


@torch.no_grad()
def eval_acc(model, head, loader, device):
    model.eval(); head.eval()
    correct, total, skipped = 0, 0, 0
    for batch in loader:
        if "x" not in batch:
            skipped += 1
            continue
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        out = model(batched_data=batch, perturb=None)
        preds = head(out).argmax(dim=1)
        y = batch["y"].view(-1)
        correct += (preds == y).sum().item()
        total += y.shape[0]
    if skipped:
        print(f"  (skipped {skipped} batches exceeding max_nodes)")
    return correct / max(1, total)


def load_checkpoint(ckpt_path, device):
    raw = torch.load(ckpt_path, map_location="cpu")
    cfg = ModelConfig.from_checkpoint(raw)
    cfg.dropout = cfg.attention_dropout = cfg.act_dropout = cfg.activation_dropout = 0.0
    args = cfg.to_namespace()

    encoder = GraphormerEncoder(args)
    model = GraphormerModel(args, encoder)
    model.load_state_dict(raw["model_state_dict"])
    model.to(device).eval()

    head = TaskHead(cfg.encoder_embed_dim, raw.get("num_classes", 2))
    head.load_state_dict(raw["head_state_dict"])
    head.to(device).eval()
    return model, head


def natural_sort_key(p):
    m = re.search(r'_run(\d+)', os.path.basename(p))
    return int(m.group(1)) if m else 0


DATASETS = [
    # (key, TUDataset name, max_nodes, batch_size)
    ("proteins", "PROTEINS", 512,  64),
    ("nci1",     "NCI1",     512,  64),
    ("dd",       "DD",       1024,  4),
]

ARCHS = ["standard", "alibi_ecc_spd"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="models/tudataset")
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model_dir = os.path.join(root_dir, args.model_dir)

    print(f"\nTUDataset size-generalisation eval  (device={device})")
    print(f"{'Config':<35} {'Val':<20} {'Test':<20} {'Gap'}")
    print("-" * 85)

    for ds_key, ds_name, max_nodes, bs in DATASETS:
        coll = make_collator(max_nodes)
        val_loader  = DataLoader(TUDatasetSizeGen(ds_name, split="val"),
                                 batch_size=bs, shuffle=False, collate_fn=coll)
        test_loader = DataLoader(TUDatasetSizeGen(ds_name, split="test"),
                                 batch_size=bs, shuffle=False, collate_fn=coll)

        for arch in ARCHS:
            pattern = os.path.join(model_dir, ds_key, f"{arch}_run*_best.pt")
            ckpts = sorted(glob.glob(pattern), key=natural_sort_key)
            val_accs, test_accs = [], []
            for ckpt in ckpts:
                model, head = load_checkpoint(ckpt, device)
                val_accs.append(eval_acc(model, head, val_loader, device))
                test_accs.append(eval_acc(model, head, test_loader, device))

            name = f"{ds_key}/{arch}"
            if val_accs:
                vm, vs = np.mean(val_accs), np.std(val_accs)
                tm, ts = np.mean(test_accs), np.std(test_accs)
                print(f"{name:<35} {vm:.3f}+/-{vs:.3f}          {tm:.3f}+/-{ts:.3f}          {vm-tm:+.3f}")
            else:
                print(f"{name:<35} no checkpoints found")


if __name__ == "__main__":
    main()
