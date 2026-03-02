# A method to do OOD evaluation

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from types import SimpleNamespace

# Path setup
_this_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(_this_dir)
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "fairseq"))

# Added the spearman r
from scipy.stats import spearmanr

from graphormer.models.graphormer import GraphormerModel, GraphormerEncoder
from graphormer.data.collator import collator
from graphormer.modules.multihead_attention import MultiheadAttention
from data.pure_tasks import get_pure_dataset
from config import ModelConfig

# Code similar to train_toy.py for loading models with hooks
def install_zero_qkt_hooks(model):
    """Zero q_proj/k_proj -> bias-only pathway."""
    hooks = []
    def zero_output_hook(module, inp, out):
        return torch.zeros_like(out)
    for module in model.modules():
        if isinstance(module, MultiheadAttention):
            hooks.append(module.q_proj.register_forward_hook(zero_output_hook))
            hooks.append(module.k_proj.register_forward_hook(zero_output_hook))
    return hooks


def install_zero_bias_hooks(model):
    """Zero attn_bias -> qkt_only pathway."""
    hooks = []
    def make_hook():
        def hook(m, args, kwargs):
            if kwargs.get('attn_bias') is not None:
                kwargs['attn_bias'] = torch.zeros_like(kwargs['attn_bias'])
            return args, kwargs
        return hook
    for module in model.modules():
        if isinstance(module, MultiheadAttention):
            hooks.append(module.register_forward_pre_hook(make_hook(), with_kwargs=True))
    return hooks

# Copy from train_toy.py for reconstructing heads
class TaskHead(nn.Module):
    def __init__(self, dim, out_dim, readout="cls", n_nodes=12, no_cls=False):
        super().__init__()
        self.proj = nn.Linear(dim, out_dim)
        self.readout = readout
        self.n_nodes = n_nodes
        self.no_cls = no_cls

    def forward(self, x):
        start = 0 if self.no_cls else 1
        if self.readout == "node":
            return self.proj(x[:, start:start + self.n_nodes, :])
        elif self.readout == "mean_pool":
            node_repr = x[:, start:start + self.n_nodes, :]
            return self.proj(node_repr.mean(dim=1))
        else:  # "cls"
            return self.proj(x[:, 0, :])

def load_model(ckpt_path, device):
    raw = torch.load(ckpt_path, map_location="cpu")

    model_cfg = ModelConfig.from_checkpoint(raw)
    # Zero dropout for deterministic eval
    model_cfg.dropout = 0.0
    model_cfg.attention_dropout = 0.0
    model_cfg.act_dropout = 0.0
    model_cfg.activation_dropout = 0.0
    model_args = model_cfg.to_namespace()

    encoder = GraphormerEncoder(model_args)
    model = GraphormerModel(model_args, encoder)
    model.load_state_dict(raw["model_state_dict"])
    model.to(device).eval()

    out_dim = raw.get("num_classes", 1)
    readout = raw.get("readout", "cls")
    no_cls = raw.get("no_cls", False)
    n_nodes = raw.get("n_nodes") or 16
    head = TaskHead(
        dim=model_cfg.encoder_embed_dim,
        out_dim=out_dim,
        readout=readout,
        n_nodes=n_nodes,
        no_cls=no_cls,
    )
    head.load_state_dict(raw["head_state_dict"])
    head.to(device).eval()

    pathway_mode = raw.get("pathway_mode", "full")
    if pathway_mode == "bias_only":
        install_zero_qkt_hooks(model)
    elif pathway_mode == "qkt_only":
        install_zero_bias_hooks(model)

    return model, head, model_args, pathway_mode

def make_collator(model_args, max_node_override=None):
    max_node = max_node_override if max_node_override is not None else model_args.max_nodes
    def graphormer_collator(items):
        obj_items = []
        for i, item in enumerate(items):
            obj = SimpleNamespace(**item)
            obj.idx = i
            obj_items.append(obj)
        return collator(
            obj_items,
            max_node=max_node,
            multi_hop_max_dist=model_args.multi_hop_max_dist,
            spatial_pos_max=model_args.spatial_pos_max,
        )
    return graphormer_collator

# Evaluation scripts
@torch.no_grad()
def eval_mae(model, head, dataset, model_args, device, batch_size=64,
             max_node_override=None):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=make_collator(model_args, max_node_override=max_node_override),
    )

    abs_errors = []
    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        out = model(batched_data=batch, perturb=None)
        logits = head(out)
        y = batch["y"].view(-1).float()
        abs_errors.append((logits.squeeze(-1) - y).abs())
    return torch.cat(abs_errors).mean().item()

@torch.no_grad()
def eval_accuracy(model, head, dataset, model_args, device, batch_size=64,
                  max_node_override=None):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=make_collator(model_args, max_node_override=max_node_override),
    )
    correct = 0
    total = 0
    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        out = model(batched_data=batch, perturb=None)
        logits = head(out)
        preds = logits.argmax(dim=1)
        y = batch["y"].view(-1)
        correct += (preds == y).sum().item()
        total += y.shape[0]
    return correct / max(1, total)

@torch.no_grad()
def eval_spearman(model, head, dataset, model_args, device, batch_size=64,
                  max_node_override=None):
    """Spearman rank correlation between predictions and targets."""
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=make_collator(model_args, max_node_override=max_node_override),
    )
    all_preds, all_targets = [], []
    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        out = model(batched_data=batch, perturb=None)
        logits = head(out)
        all_preds.append(logits.squeeze(-1).cpu())
        all_targets.append(batch["y"].view(-1).float().cpu())
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    rho, _ = spearmanr(preds, targets)
    return rho

def make_dataset(task, num_nodes, num_samples, seed, target_metric="apl", cache_path=None):
    # Dataset helper

    kwargs = dict(num_nodes=num_nodes, num_samples=num_samples, seed=seed)
    if task == "semantic":
        kwargs["feature_dim"] = 2
    elif task == "positional" and target_metric != "apl":
        kwargs["target_metric"] = target_metric
    if cache_path:
        kwargs["cache_path"] = cache_path
    return get_pure_dataset(task, **kwargs)

TASK_CONFIG = {
    "positional": {"metric": "mae", "default_dir": "models/positional/2L"},
    "semantic":   {"metric": "accuracy", "default_dir": "models/semantic/2L"},
}

def main():
    parser = argparse.ArgumentParser(
        description="Unified OOD eval: train x test matrix across graph sizes"
    )
    parser.add_argument("--task", type=str, required=True,
                        choices=list(TASK_CONFIG.keys()),
                        help="Task type")
    parser.add_argument("--conditions", type=str, nargs="+", default=["baseline"],
                        help="Conditions to evaluate (default: baseline)")
    parser.add_argument("--n_values", type=int, nargs="+", default=[8, 16, 32, 64],
                        help="Graph sizes for full n_train x n_test matrix")
    parser.add_argument("--model_dir", type=str, default="",
                        help="Model root dir; loads {model_dir}/n{N}/{condition}.pt")
    parser.add_argument("--target_metric", type=str, default="apl",
                        choices=["apl", "global_efficiency", "ge_sqrt_n"],
                        help="Target metric (positional task only)")
    parser.add_argument("--save_results", type=str, default="",
                        help="Output path (default: results/ood_{task}.pt)")
    parser.add_argument("--ood_samples", type=int, default=500,
                        help="Samples per test dataset")
    parser.add_argument("--extra_test_n", type=int, nargs="*", default=[],
                        help="Extra test-only sizes (no checkpoint needed); "
                             "loaded from --ood_dataset_dir if caches are present")
    parser.add_argument("--ood_dataset_dir", type=str, default="",
                        help="Dir containing pre-generated OOD caches named n{N}_ood.pt")
    parser.add_argument("--test_dataset_dir", type=str, default="",
                        help="Dir with pre-existing test caches; tries n{N}_test.pt then n{N}.pt")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Base batch size; auto-scaled down quadratically for large n_test "
                             "(default: 128 so n=256→8, n=512→2 at reference n=64)")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--best", action="store_true",
                        help="Load _best.pt checkpoints instead of last .pt")
    args = parser.parse_args()

    task_cfg = TASK_CONFIG[args.task]
    metric = task_cfg["metric"]
    eval_fn = eval_accuracy if metric == "accuracy" else eval_mae

    if not args.save_results:
        variant = "_best" if args.best else ""
        args.save_results = os.path.join(root_dir, "results", f"ood_{args.task}{variant}.pt")
    if not args.model_dir:
        args.model_dir = os.path.join(root_dir, task_cfg["default_dir"])

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    # Ordered, deduped union of training sizes and OOD-only test sizes
    seen = {}
    for n in args.n_values + args.extra_test_n:
        seen[n] = None
    all_test_n = list(seen)

    print(f"Device: {device}")
    print(f"Task: {args.task} ({metric})")
    print(f"Conditions: {args.conditions}")
    print(f"Train sizes: {args.n_values}")
    if args.extra_test_n:
        print(f"OOD test sizes: {args.extra_test_n}")
    print(f"All test sizes: {all_test_n}")
    print(f"Model dir: {args.model_dir}")

    # --- Build test datasets (one per n_test, random fresh seed) ---
    print(f"\nBuilding test datasets ({args.ood_samples} samples each")
    test_datasets = {}
    for n in all_test_n:
        cache_path = None

        if args.test_dataset_dir and cache_path is None:
            for suffix in ("_test.pt", ".pt"):
                candidate = os.path.join(args.test_dataset_dir, f"n{n}{suffix}")
                if os.path.exists(candidate):
                    cache_path = candidate
                    break

        if args.ood_dataset_dir and cache_path is None and n in args.extra_test_n:
            candidate = os.path.join(args.ood_dataset_dir, f"n{n}_ood.pt")
            if os.path.exists(candidate):
                cache_path = candidate
        source = f"cache: {cache_path}" if cache_path else "generating..."
        print(f"  n={n} ({source}) ...", end=" ", flush=True)
        test_datasets[n] = make_dataset(
            args.task, num_nodes=n, num_samples=args.ood_samples,
            seed=77777, target_metric=args.target_metric,
            cache_path=cache_path,
        )
        print("done")

    matrices = {c: {} for c in args.conditions}
    correlations = {c: {} for c in args.conditions} if metric == "mae" else None

    for condition in args.conditions:
        print(f"\n--- Condition: {condition} ---")
        for n_train in args.n_values:
            suffix = "_best" if args.best else ""
            ckpt_path = os.path.join(args.model_dir, f"n{n_train}", f"{condition}{suffix}.pt")
            if not os.path.exists(ckpt_path):
                print(f"  Skip n_train={n_train}: {ckpt_path} not found")
                continue

            print(f"  Loading n_train={n_train} from {ckpt_path}")
            model, head, model_args, pathway_mode = load_model(ckpt_path, device)

            matrices[condition][n_train] = {}
            if correlations is not None:
                correlations[condition][n_train] = {}
            for n_test in all_test_n:
                max_node_override = max(n_test + 2, model_args.max_nodes)
                # Quadratic auto-scaling: keep memory roughly constant (ref = n=64)
                effective_bs = max(1, args.batch_size * 64 * 64 // (n_test * n_test))
                val = eval_fn(
                    model, head, test_datasets[n_test], model_args, device,
                    batch_size=effective_bs,
                    max_node_override=max_node_override,
                )
                matrices[condition][n_train][n_test] = val
                tag = "(ID)" if n_test == n_train else "(OOD)" if n_test not in args.n_values else ""
                if metric == "accuracy":
                    print(f"    n_test={n_test:<4} {tag:>5}: {val:.2%}  [bs={effective_bs}]")
                else:
                    rho = eval_spearman(
                        model, head, test_datasets[n_test], model_args, device,
                        batch_size=effective_bs,
                        max_node_override=max_node_override,
                    )
                    correlations[condition][n_train][n_test] = rho
                    print(f"    n_test={n_test:<4} {tag:>5}: MAE={val:.4f}  ρ={rho:+.3f}  [bs={effective_bs}]")

    metric_label = "Accuracy" if metric == "accuracy" else "MAE"
    for condition in args.conditions:
        mat = matrices[condition]
        ns_avail = [n for n in args.n_values if n in mat]
        if not ns_avail:
            continue
        print(f"\n{'='*60}")
        print(f"{args.task.title()} {condition} — {metric_label}")
        print(f"{'':>12}", end="")
        for n_test in all_test_n:
            ood_marker = "+" if n_test in args.extra_test_n else ""
            print(f"  n_test={n_test}{ood_marker:<2}", end="")
        print()
        print(f"{'-'*60}")
        for n_train in ns_avail:
            print(f"n_train={n_train:<4}", end="")
            for n_test in all_test_n:
                val = mat[n_train].get(n_test)
                if val is not None:
                    marker = "*" if n_test == n_train else " "
                    if metric == "accuracy":
                        print(f"  {val*100:5.1f}%{marker}    ", end="")
                    else:
                        print(f"  {val:.4f}{marker}     ", end="")
                else:
                    print(f"  {'N/A':<10}", end="")
            print()
        print(f"{'='*60}")
        print("* = in-distribution  + = OOD-only (no checkpoint at this size)")

    if correlations is not None:
        for condition in args.conditions:
            corr = correlations[condition]
            ns_avail = [n for n in args.n_values if n in corr]
            if not ns_avail:
                continue
            print(f"\n{'='*60}")
            print(f"{args.task.title()} {condition} — Spearman ρ")
            print(f"{'':>12}", end="")
            for n_test in all_test_n:
                ood_marker = "+" if n_test in args.extra_test_n else ""
                print(f"  n_test={n_test}{ood_marker:<2}", end="")
            print()
            print(f"{'-'*60}")
            for n_train in ns_avail:
                print(f"n_train={n_train:<4}", end="")
                for n_test in all_test_n:
                    val = corr[n_train].get(n_test)
                    if val is not None:
                        marker = "*" if n_test == n_train else " "
                        print(f"  {val:+.3f}{marker}     ", end="")
                    else:
                        print(f"  {'N/A':<10}", end="")
                print()
            print(f"{'='*60}")
            print("* = in-distribution  + = OOD-only (no checkpoint at this size)")

    # Save it for plotting
    save_dict = {
        'task': args.task,
        'n_values': args.n_values,
        'extra_test_n': args.extra_test_n,
        'all_test_n': all_test_n,
        'conditions': args.conditions,
        'ood_samples': args.ood_samples,
        'metric': metric,
        'matrices': matrices,
        'checkpoint_variant': 'best' if args.best else 'last',
    }
    if correlations is not None:
        save_dict['correlations'] = correlations
    if args.task == "positional":
        save_dict['target_metric'] = args.target_metric
    os.makedirs(os.path.dirname(args.save_results), exist_ok=True)
    torch.save(save_dict, args.save_results)
    print(f"\nResults saved to {args.save_results}")

if __name__ == "__main__":
    main()
